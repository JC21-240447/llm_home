# Day 12.2 — training.py (updated) — What Changed + Line-by-Line Explanation (English)

> **This is the UPDATED version of `training.py`** that you first saw on Day 9.2.
> - **Most of the file is identical.** Its full line-by-line explanation is in **`day9.2_training_explained_EN.md`** (and the JA file): the config, the learning-rate schedule, weight decay groups, evaluation, checkpoints, Ctrl+C handling, resume, logging.
> - **New in Day 12: early stopping + `status.json`.** This file explains every changed line.
>
> **How the five Day 12 files fit together:** `day12_train_main.py` (uses early stopping), `day12_monitor.py` (shows the early-stop counter and the status), `generation.py` (12.1), **`training.py` (12.2, this file)**, `test_training.py` (12.3, tests of the new behaviour).

---

## 0. The big picture (read this first)

### The problem

In Day 11 the best validation loss often came **early** (steps 125–225) and the loss rose afterwards (**overfitting**). Training the full schedule wastes time, and the *last* checkpoint is worse than the *best* one.

### Early stopping

> If the validation loss has **not improved for N steps** (the **patience**), **stop training**. `best.pt` already holds the best model, so nothing is lost.

```
validation loss:  3.7 → 2.9 → 2.4 → 2.2 (best, step 120) → 2.25 → 2.3 → 2.31 → 2.4 ...
                                              ↑ best_step                 └── 100 steps without improvement → STOP
```
Example with the Day 12 settings: validation every 10 steps, patience 100 → stop after **10 evaluations in a row without a new best**.

### `status.json` — "how did this run end?"

When a run ends, it writes a small file `status.json`:

| `status` | Meaning |
|---|---|
| `finished` | reached `max_steps` normally |
| `early_stopped` | stopped by early stopping |
| `stopped` | stopped by Ctrl+C or `stop_after` (can be resumed) |

The Day 12 script reads this file to decide whether a seed is **done**, or must be **resumed** (instead of parsing the log like Day 11 did).

---

## 1. Unchanged parts (pointer table)

| Part | Explained in |
|---|---|
| `lr_at`, `param_groups`, `pick_device`, `pick_precision`, `autocast_ctx`, `sha256_file` | Day 9.2 §4 |
| `evaluate`, `greedy_continue`, `save_checkpoint`, `load_checkpoint`, `_StopFlag`, `_truncate_after`, `_fmt` | Day 9.2 §4 |
| checking settings vs data, building the model/optimizer, the log writer, the main loop structure | Day 9.2 §5 |

---

## 2. What changed, change by change

### 2.1 The docstring

A new paragraph: *"Early stopping (Day 12, optional): with `early_stop_patience = N`, training ends when the validation loss has not improved for N steps; `best.pt` keeps the best model. `status.json` records how the run ended: finished, early_stopped, or stopped (Ctrl+C, resume later)."*

### 2.2 A new setting in `TrainConfig`

```python
    early_stop_patience: int | None = None  # Day 12: stop when validation has not improved for this many steps
```
- **`None` = early stopping is OFF** (the default), so Day 9 and Day 11 behave exactly as before. A number turns it on.
- It sits next to `stop_after`. *Note:* it is **not** in `RESUME_FREE_KEYS`, so a resumed run must use the **same patience** as the first run (otherwise "resume refused").

### 2.3 A new run also deletes the old `status.json`

```python
        for p in (last_path, best_path, log_path, samples_path, run_dir / "status.json"):
            p.unlink(missing_ok=True)
```
- When starting a **new** run in a folder, the list of leftover files to delete now also includes `status.json`, so an old "finished" label can never be mistaken for the new run's.
- (A **resumed** run does *not* delete it: see "Small things" below.)

### 2.4 Remember *where* the best point was: `best_step`

```python
    step, best_val, elapsed_before, tokens_seen = 0, float("inf"), 0.0, 0
    best_step = 0
```
- A new counter `best_step` — the **step at which the lowest validation loss happened** (needed to count "steps without improvement").

```python
        step, best_val = st["step"], st["best_val"]
        best_step = st.get("best_step", step)
```
- **On resume**, restore it from the checkpoint. `st.get("best_step", step)` — `dict.get(key, default)`: if the checkpoint is **old** (made before Day 12, no `best_step` inside), fall back to the current `step`. So older checkpoints still load.

```python
            "model_config": mcfg.to_dict(), "train_config": cfg.to_dict(), "best_val": best_val, "best_step": best_step,
```
- `state_for_save` (what goes into a checkpoint) now also stores **`best_step`**. This is essential: without it, a resumed run would forget when the best point was, and early stopping would behave differently from a non-stop run (Day 12.3 tests exactly this).

### 2.5 Update `best_step` at every evaluation

```python
    def checkpoint_and_sample(lr: float) -> float:
        nonlocal best_val, best_step
        val = evaluate(...)
        improved = val < best_val
        best_val = min(best_val, val)
        if improved:
            best_step = step
```
- `nonlocal best_val, best_step` — the inner function may now **change** both outer variables (same `nonlocal` idea as Day 9.2).
- If this evaluation is a **new best** (`improved`), remember the current `step` as `best_step`. (The rest of the function — logging, samples, saving `last.pt` / `best.pt` — is unchanged.)

### 2.6 Two flags in the main loop

```python
    stopped_early = False
    early_stopped = False
```
- `stopped_early` (existing) = stopped by Ctrl+C / `stop_after` (resume later). **`early_stopped` (new)** = stopped by early stopping (the run is *complete*, not to be resumed).

### 2.7 The early-stopping check (the heart of the change)

```python
                if (cfg.early_stop_patience is not None and step % cfg.eval_every == 0 and step < cfg.max_steps
                        and step - best_step >= cfg.early_stop_patience):
                    early_stopped = True
                    print(f"Early stop at step {step}: no improvement for {step - best_step} steps "
                          f"(best validation loss {best_val:.4f} at step {best_step}; best.pt keeps it).")
                    break
```
This sits at the end of each step, **after** the usual evaluation. All four conditions must be true:

| Condition | Why |
|---|---|
| `cfg.early_stop_patience is not None` | early stopping must be **switched on** |
| `step % cfg.eval_every == 0` | only at an **evaluation step** (so `best_step` is up to date) |
| `step < cfg.max_steps` | at the last step the run ends normally anyway |
| `step - best_step >= cfg.early_stop_patience` | **at least N steps since the best point** |

- If all are true: set `early_stopped = True`, print a message (how many steps without improvement, the best loss and where) and **`break`** out of the loop.
- Because this check comes **after** `checkpoint_and_sample`, `last.pt` has already been saved at this step. `best.pt` is the best model.
- *Order in the loop:* the Ctrl+C / `stop_after` block comes first (if it triggers, it `break`s before reaching here).

### 2.8 After the loop: write `status.json`

```python
    status = "stopped" if stopped_early else ("early_stopped" if early_stopped else "finished")
    (run_dir / "status.json").write_text(json.dumps({"status": status, "step": step, "best_step": best_step,
                                                      "best_val": best_val}, indent=2) + "\n",
                                         encoding="utf-8", newline="\n")
```
- Decide the **status** with a nested conditional: Ctrl+C/`stop_after` → `"stopped"`; else early stopping → `"early_stopped"`; else `"finished"`.
- Save `{status, step, best_step, best_val}` as JSON (LF line endings, as always).
- This code is **after** the `try/finally`, so it runs for normal ends. If training **crashes** (for example the "loss became NaN" `SystemExit`) or you press Ctrl+C **twice**, no `status.json` is written — and the Day 12 script treats "has `last.pt` but no status" as **unfinished** (resume).

### 2.9 A richer return value

```python
    return {"run_dir": str(run_dir), "step": step, "max_steps": cfg.max_steps, "best_val": best_val,
            "best_step": best_step, "early_stopped": early_stopped, "status": status,
            "stopped_early": stopped_early, "device": device_name, "precision": precision, "last_lr": last_lr,
            "elapsed_s": elapsed_before + time.perf_counter() - t_start}
```
- The summary dictionary gains **`best_step`**, **`early_stopped`** and **`status`**. (Old keys are unchanged, so Day 9 and Day 11 scripts keep working.)

---

## 3. Small things worth knowing

- **A resumed run keeps the old `status.json` until it ends.** The "new run" branch deletes it, the "resume" branch does not. So while a *resumed* run is training, a monitor can still show `[stopped]` from the previous stop; it updates when the run ends.
- **Early stopping looks only at evaluation steps**, so with `eval_every = 10` and `patience = 100` the run stops at the first multiple of 10 where it is ≥ 100 steps past the best.
- **The best model is never lost**: `best.pt` is saved at every improvement, regardless of stopping.
- **The cosine learning-rate schedule still follows `max_steps`**: an early-stopped run simply ends before the schedule reaches its minimum.
- **Old checkpoints still load** (`best_step` falls back to the current step), but then "steps without improvement" starts counting from there.
- **Day 9 and Day 11 are unaffected**: with `early_stop_patience = None` nothing changes except that a `status.json` file is now written at the end.

## 4. Quick summary in one paragraph

`training.py` gains optional early stopping: a new setting `early_stop_patience`, a `best_step` counter that is updated at each evaluation and saved in checkpoints (so resuming is exact), a check after each evaluation step that stops the loop when the validation loss has not improved for that many steps, and a `status.json` file (`finished` / `early_stopped` / `stopped`) written at the end so scripts can tell how a run ended; everything else is as in Day 9.2.
