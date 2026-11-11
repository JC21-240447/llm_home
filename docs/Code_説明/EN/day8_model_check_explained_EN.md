# Day 8 — day8_model_check.py — Simple Line-by-Line Explanation (English)

> **How the three Day 8 files fit together**
> - **Day 8.1 `model.py`** — the model itself (**read it first**; this script builds and runs it).
> - **Day 8 (this file) `day8_model_check.py`** — the *driver*: builds Model A at three sizes, times it, checks it on real data, saves configs, reports.
> - **Day 8.2 `test_model.py`** — automatic tests for the model.

---

## 0. The big picture (read this first)

Before spending hours on training (Day 9), you want to be sure of three things:

1. **How big is each candidate model, and how fast is it on this GPU?**
2. **Is the model wired correctly?** (a wrongly wired model would "train" but learn nothing)
3. **Do we have enough data for each size?**

This script is a **pre-flight check** for an aircraft:

| Check | What happens | Why it proves something |
|---|---|---|
| **Sizes** | For tiny / small / medium: count parameters, compute *training tokens per parameter* | Tells how "hungry" each model is for data |
| **Speed & memory** | One timed training step (forward + backward) with the real block size | Tells if the model fits and how long training will take |
| **Initial loss** | Before training, loss ≈ `ln(vocabulary size)` | The model knows nothing yet, so it should guess uniformly |
| **Two attentions agree** | Manual version = fast version | Proves the fast code does the same maths as the readable code |
| **Memorise one batch** | Loss on one real batch falls close to 0 | Proves model + loss + gradients + optimizer work together |

Then it saves the three configs (`configs/model_*.json`) for Day 9 and Day 11, writes `docs/DAY8_MODEL_REPORT.md`, and adds a note to `run_log.md`.

Run it:

```
.venv\Scripts\python.exe scripts\day8_model_check.py --machine home
```

---

## 1. Words you need to know (mini dictionary)

| Word | Simple meaning |
|---|---|
| **Forward pass** | Data goes through the model to produce predictions and the loss. |
| **Backward pass** | The model works out **gradients**: for every parameter, "which way should it change to lower the loss?" |
| **Training step** | One forward + backward (+ an optimizer update in real training). |
| **Optimizer (AdamW)** | The rule that actually changes the parameters using the gradients. |
| **Batch** | Several sequences processed together. |
| **Overfit / memorise** | Learn a small set of examples by heart. Here used on purpose as a **test**: a correct model *must* be able to do it. |
| **Training tokens per parameter** | `training tokens ÷ parameters`. A rough sign of how much data there is per learnable number. |
| **Tokens per second** | Speed: how many tokens are processed each second. |
| **Peak GPU memory** | The most GPU memory used at any moment. |
| **bf16 / fp16 / fp32** | Number formats (see Day 1). bf16 needs a newer GPU (compute capability 8+). |
| **Autocast** | Automatic mixed precision (see Day 1). |
| **`torch.device`** | Says where the computation happens: `cuda` (GPU) or `cpu`. |

---

## 2. The docstring

- English + Japanese, the 4 steps, usage, and the **server note**: parameter counts are the same everywhere; speed and memory depend on the GPU. **On a GPU without bf16 (e.g. RTX 2060) the timing uses fp16 and the memorisation check uses fp32** — the script chooses by itself.

---

## 3. Imports and setup

```python
from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
```
- All as in earlier days, plus:
  - `nullcontext` — a "do nothing" `with` block (see `autocast` below);
  - `numpy as np` (as in Day 7.1) and `torch` (as in Day 8.1).

```python
REPO_ROOT: Path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.bpe_tokenizer import ByteBPETokenizer  # noqa: E402
from src.model import PRESETS, ModelA, count_parameters, expected_parameters, preset_config  # noqa: E402

SCRIPT_VERSION: str = "day8-v1"
```
- Same `sys.path` trick as Days 6–7. Imports the tokenizer (for the vocabulary size) and, from **Day 8.1**, the model, the presets and the counting helpers.

---

## 4. `SizeReport` dataclass

```python
@dataclass
class SizeReport:
    name: str
    n_layer: int
    n_head: int
    d_model: int
    total: int
    embedding: int
    non_embedding: int
    tokens_per_param: float
    ms_per_step: float
    tokens_per_sec: float
    peak_mem_mb: float
```
- One "result card" per model size (same idea as Day 6/7): preset name, layers/heads/width, parameter counts (total / embedding / non-embedding), training tokens per parameter, milliseconds per step, tokens per second, peak GPU memory in MB.

---

## 5. Helper functions

### 5.1 `pick_device`

```python
def pick_device(choice: str) -> torch.device:
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if choice == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda was requested but CUDA is not available")
    return torch.device(choice)
```
- `auto` → GPU if available, otherwise CPU. If you **ask for** `cuda` but there is none → stop with a clear message (instead of silently using the CPU). Otherwise use what you asked for.

### 5.2 `precision`

```python
def precision(device: torch.device) -> tuple[str, torch.dtype | None]:
    """(label, autocast dtype). bf16 where supported, fp16 on older GPUs, fp32 on the CPU."""
    if device.type != "cuda":
        return "fp32", None
    if torch.cuda.is_bf16_supported():
        return "bf16", torch.bfloat16
    return "fp16", torch.float16
```
- Chooses the number format: **CPU → fp32** (no autocast, `None`); **GPU with bf16 → bf16**; **older GPU → fp16** (this is the Day 1 "recommended precision" idea).
- *Caution:* on some PyTorch versions `is_bf16_supported()` can answer "yes" for an older GPU by **emulating** bf16 (slow; see Day 1's `native_bf16` vs reported). The report prints the **precision actually used**, so check it: an RTX 2060 should normally show fp16.

### 5.3 `autocast`

```python
def autocast(device: torch.device, dtype: torch.dtype | None):
    return torch.autocast(device_type=device.type, dtype=dtype) if dtype is not None else nullcontext()
```
- Returns a ready-to-use `with` block: real **autocast** (Day 1) when a dtype is given, or `nullcontext()` (does nothing) for full 32-bit. This lets the same code `with autocast(...):` work for both.

### 5.4 `real_batch`

```python
def real_batch(tokens: np.ndarray, batch: int, length: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """batch windows of length+1 tokens from fixed random positions: x = first length, y = shifted by one."""
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(tokens) - length - 1, size=batch)
    win = np.stack([np.asarray(tokens[s:s + length + 1], dtype=np.int64) for s in starts])
    w = torch.from_numpy(win)
    return w[:, :-1], w[:, 1:]
```
- Cuts **real training windows** out of the Day 7 token stream (the `train.bin` memory map).
- `np.random.default_rng(seed)` — a seeded random generator (NumPy's version); `rng.integers(0, N, size=batch)` picks `batch` random **start positions**.
- For each start `s`, take `length + 1` tokens `tokens[s : s+length+1]`, convert to 64-bit integers (`np.asarray(..., dtype=np.int64)` also copies them out of the memory map). `np.stack` stacks them into a table `(batch, length+1)`.
- `torch.from_numpy` makes a PyTorch tensor.
- **`x = w[:, :-1]`** (the first `length` tokens) and **`y = w[:, 1:]`** (the same window shifted by one). At every position the model is asked: "given tokens up to here, what is the next token?" — the answer is the `y` value. (A window may cross from one question into the next in the stream; that is fine.)

### 5.5 `time_training_step`

```python
def time_training_step(model, x, y, device, dtype, repeats) -> tuple[float, float]:
    """Average milliseconds of one forward + backward pass, and the peak GPU memory in MB."""
    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def step() -> None:
        with autocast(device, dtype):
            _, loss = model(x, y)
        loss.backward()
        model.zero_grad(set_to_none=True)
```
- `model.train()` — training mode (matters for dropout). Reset the GPU's "peak memory" counter so the peak we read is for **this** model only.
- `step()` — one forward pass (under autocast) giving the loss, `loss.backward()` (the backward pass: fills every parameter's `.grad`), then `zero_grad(set_to_none=True)` to throw the gradients away (so memory is freed and the next step starts clean). **No optimizer update** — it is only a speed test.

```python
    step()  # warm-up (memory allocation, kernel selection)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for _ in range(repeats):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    ms = (time.perf_counter() - t0) * 1000 / repeats
    peak = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else float("nan")
    return ms, peak
```
- **Warm-up** one step first (the first step is slow: memory allocation, choosing GPU kernels). Same warm-up + `synchronize` idea as Day 1's `bench_matmul` (the GPU works in the background, so wait until it has really finished before reading the clock).
- Time `repeats` steps (10 by default), convert to **milliseconds per step**.
- `peak` — the maximum GPU memory used, in MB (`/ 2**20` converts bytes to MB), or `NaN` ("not a number") on the CPU.

### 5.6 `memorise_one_batch`

```python
def memorise_one_batch(vocab, block, x, y, device, dtype, steps, seed) -> list[float]:
    torch.manual_seed(seed)
    model = ModelA(preset_config("tiny", vocab, block)).to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    use = dtype if dtype == torch.bfloat16 else None   # fp16 would need a gradient scaler: use fp32 instead
    losses: list[float] = []
    for _ in range(steps):
        with autocast(device, use):
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses
```
- The **real training loop**, in miniature, on **one fixed batch** (so the model *can* memorise it):
  - fixed seed → a fresh tiny model on the device, in train mode;
  - `AdamW` optimizer: learning rate `3e-3`, no weight decay;
  - `use = dtype if dtype == torch.bfloat16 else None` — **only bf16 is used**; fp16 would need a gradient scaler (Day 1), so on an fp16 GPU this check simply runs in fp32 (the docstring mentions this).
  - Each step: forward → loss; `zero_grad` (clear old gradients); `backward` (compute gradients); `opt.step()` (**update the parameters**); remember `loss.item()` (the loss as a plain number).
- Returns the list of 300 losses (default). A correct setup must make it drop towards 0.

### 5.7 `write_report`, `mem_text`, `_milestones`

```python
def write_report(path, a, device_name, prec, vocab, train_tokens, reports, init_loss, attn_diff, losses) -> None:
    lines = [ ... ]
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
```
- Writes `docs/DAY8_MODEL_REPORT.md` (same pattern as other days). Sections:

| Section | Content |
|---|---|
| Header | version, date, machine label, device, **precision for timing**, PyTorch/Python versions, vocabulary, training tokens, block size, timing batch |
| **Sizes** | A table, one row per preset: layers, heads, width, parameters (total / embedding / non-embedding), **training tokens per parameter**, ms per step, tokens per second, peak GPU memory. Notes: one step = forward + backward (no optimizer); embedding and output layer share one matrix so it is counted once; peak memory shows `-` on the CPU; "the size is chosen on Day 11 by validation loss" |
| **Checks on real token windows** | Initial loss vs `ln(vocab)`; largest difference between manual and fast attention; memorising one batch (first loss → last loss); a small table of the loss at 5 milestones |

- `*(... for r in reports)` unpacks the table rows (same trick as before).

```python
def mem_text(mb: float) -> str:
    return "-" if math.isnan(mb) else f"{mb:,.0f}"
```
- Shows `-` for `NaN` (CPU), otherwise MB with thousands commas.

```python
def _milestones(n: int) -> list[int]:
    return sorted({max(1, n * k // 5) for k in range(1, 6)})
```
- Five evenly spaced step numbers: for 300 steps → `{60, 120, 180, 240, 300}` (`n*k//5`, at least 1; the set removes duplicates for very small `n`). Used to print a few points of the loss curve instead of all 300.
- (`write_report` uses `_milestones` and `mem_text`, which are defined below it; that works because the functions are only *called* later.)

### 5.8 `append_run_log`

```python
def append_run_log(log_path, a, device_name, prec, reports, init_loss, attn_diff, losses) -> None:
    body = [ ... ]
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"\n## {datetime.now():%Y-%m-%d %H:%M} | {a.machine} | Day 8 model check\n\n" + "\n".join(body) + "\n")
```
- Same style as Day 7: bullet lines (machine, device, precision, script/PyTorch/Python versions, one line per size with parameters, ms/step, tokens/s, peak memory, and the three check results) appended to `run_log.md`.

---

## 6. `parse_args`

```python
ap.add_argument("--machine", choices=["home", "server", "school"], required=True)
ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
ap.add_argument("--block-size", type=int, default=256)
ap.add_argument("--batch", type=int, default=32, help="batch size for the timing step")
ap.add_argument("--repeats", type=int, default=10, help="timed steps per size")
ap.add_argument("--overfit-steps", type=int, default=300)
ap.add_argument("--overfit-batch", type=int, default=4)
ap.add_argument("--overfit-len", type=int, default=64)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--tokenizer", ...)  ap.add_argument("--tokens-dir", ...)  ap.add_argument("--configs-dir", ...)
ap.add_argument("--report", ...)  ap.add_argument("--run-log", ...)  ap.add_argument("--no-log", action="store_true")
```
- Same `argparse` idea as before. The new numeric options:
  - `--device` — `auto` / `cuda` / `cpu`;
  - `--block-size` 256 — sequence length (also the model's `block_size`);
  - `--batch` 32 — batch size for the timing test (**lower it if the GPU runs out of memory**, e.g. on the 6 GB RTX 2060);
  - `--repeats` 10 — timed steps per size;
  - `--overfit-steps` 300, `--overfit-batch` 4, `--overfit-len` 64 — the memorisation test (4 windows of 64 tokens, 300 steps);
  - `--seed` 42.
- The `Path` options (tokenizer, token files folder, configs folder, report, run log) and `--no-log` work as before.

---

## 7. `main`

### 7.1 Setup and checks of the inputs

```python
def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    a = parse_args()
    for name in ("tokenizer", "tokens_dir", "configs_dir", "report", "run_log"):
        setattr(a, name, getattr(a, name).resolve())
    meta_path = a.tokens_dir / "train_meta.json"
    if not a.tokenizer.exists():
        raise SystemExit(f"Missing {a.tokenizer}. Run Day 6 first.")
    if not meta_path.exists():
        raise SystemExit(f"Missing {meta_path}. Run Day 7 first.")
```
- Same prelude as before. Needs the tokenizer (Day 6) and `train_meta.json` (Day 7).

```python
    device = pick_device(a.device)
    prec, dtype = precision(device)
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print(f"=== Day 8 model check | machine={a.machine} | {device_name} | {prec} | PyTorch {torch.__version__} ===")
```
- Choose the device and precision; get the GPU's name (or "CPU"); print a header.

```python
    vocab = ByteBPETokenizer.load(a.tokenizer).vocab_size
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta["vocab_size"] != vocab:
        raise SystemExit("train_meta.json was made with a different tokenizer. Rerun Day 7.")
    train_tokens = int(meta["tokens"])
    tokens = np.memmap(a.tokens_dir / "train.bin", dtype=np.dtype("<u2"), mode="r")
    print(f"Vocabulary {vocab:,}; training tokens {train_tokens:,}; block size {a.block_size}")
```
- `vocab` — the vocabulary size from the **real tokenizer**.
- Read the Day 7 meta file. **Consistency check:** its `vocab_size` must equal the tokenizer's, otherwise the token file was made with another tokenizer → stop. (This is the kind of check the Day 7.1 notes said the *consumer* must do; it compares the vocabulary size, not the SHA-256.)
- `train_tokens` — number of training tokens (from the meta).
- `np.memmap(..., dtype=np.dtype("<u2"), mode="r")` — opens `train.bin` directly as a **read-only memory map** of little-endian `uint16` (the Day 7.1 format) — without loading the whole file.

### 7.2 Check 1: three sizes, parameters and one timed step

```python
    a.configs_dir.mkdir(parents=True, exist_ok=True)
    x, y = (t.to(device) for t in real_batch(tokens, a.batch, a.block_size, a.seed))
    reports: list[SizeReport] = []
    print(f"\n{'preset':<7} {'params':>11} {'tokens/param':>12} {'ms/step':>9} {'tokens/s':>11} {'peak MB':>8}")
```
- Make the configs folder. Take **one real batch** (`batch × block_size` tokens) and move `x`, `y` to the device (`.to(device)`; the generator is unpacked into two names). The **same batch** is used for all three sizes → a fair comparison.
- Print a table header (alignment codes as before).

```python
    for name in PRESETS:
        cfg = preset_config(name, vocab, a.block_size)
        cfg.save(a.configs_dir / f"model_{name}.json")
        torch.manual_seed(a.seed)
        model = ModelA(cfg).to(device)
        counts = count_parameters(model)
        if counts["total"] != expected_parameters(cfg):
            raise SystemExit(f"{name}: parameter count does not match the formula")
        ms, peak = time_training_step(model, x, y, device, dtype, a.repeats)
```
- For `tiny`, `small`, `medium` in turn:
  - build the config for the **real vocabulary** and **save it** as `configs/model_<name>.json` (for Day 9/11);
  - fix the seed, build the model and move it to the GPU;
  - **cross-check the parameter count** against the hand formula (Day 8.1 §7) — a mismatch means a bug → stop;
  - time one training step.

```python
        r = SizeReport(name, cfg.n_layer, cfg.n_head, cfg.d_model, counts["total"], counts["embedding"],
                       counts["non_embedding"], train_tokens / counts["total"], ms,
                       a.batch * a.block_size / (ms / 1000), peak)
        reports.append(r)
        print(f"{name:<7} {r.total:>11,} {r.tokens_per_param:>12.2f} {r.ms_per_step:>9.1f} "
              f"{r.tokens_per_sec:>11,.0f} {mem_text(r.peak_mem_mb):>8}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
```
- Result card:
  - **`tokens_per_param = train_tokens / total parameters`** — how many training tokens there are per learnable number;
  - **`tokens_per_sec = batch × block_size / (ms ÷ 1000)`** — tokens processed per second (tokens in a step ÷ seconds per step).
- Print the row. Then `del model` and `empty_cache()` to **free GPU memory** before the next size (so one size's memory does not distort the next).

### 7.3 Check 2: initial loss and the two attentions

```python
    torch.manual_seed(a.seed)
    tiny = ModelA(preset_config("tiny", vocab, a.block_size)).to(device).eval()
    with torch.no_grad():
        fast_logits, init = tiny(x, y)
        tiny.set_attention("manual")
        manual_logits, _ = tiny(x, y)
    init_loss = init.item()
    attn_diff = (fast_logits - manual_logits).abs().max().item()
```
- A fresh **untrained** tiny model in evaluation mode, in plain **fp32** (no autocast here).
- `torch.no_grad()` — don't track gradients (faster, no memory for them).
- Run the real batch with the **fast** attention → `fast_logits` and the loss `init`; switch to **manual** attention (`set_attention`, same weights), run again → `manual_logits`.
- `init_loss` — the loss before any training.
- `attn_diff` — the **largest absolute difference** between the two logits tensors (`.abs().max()`), as a Python number (`.item()`).

```python
    print(f"\nLoss before training: {init_loss:.3f}  (ln {vocab} = {math.log(vocab):.3f})")
    print(f"Manual vs fast attention: max difference {attn_diff:.2e}")
    if abs(init_loss - math.log(vocab)) > 0.5 or attn_diff > 1e-3:
        raise SystemExit("Check FAILED: initial loss or attention difference out of range")
    del tiny
```
- Print both. **Fail** if the initial loss is more than 0.5 away from `ln(vocab)` (e.g. a wiring error or bad initialisation) or the attentions differ by more than 0.001. (`:.2e` = scientific notation like `3.1e-07`.)
- Why `ln(vocab)`? A model guessing uniformly gives each of the `V` tokens probability `1/V`, so the loss is `−log(1/V) = ln V`. For `V = 1,674` that is about **7.42**.

### 7.4 Check 3: memorise one real batch

```python
    ox, oy = (t.to(device) for t in real_batch(tokens, a.overfit_batch, a.overfit_len, a.seed + 1))
    losses = memorise_one_batch(vocab, a.block_size, ox, oy, device, dtype, a.overfit_steps, a.seed)
    print(f"Memorise one batch ({a.overfit_batch} x {a.overfit_len}): loss {losses[0]:.3f} -> {losses[-1]:.4f}")
    for s in _milestones(len(losses)):
        print(f"   step {s:>4}: {losses[s - 1]:.4f}")
    if losses[-1] > 0.5:
        raise SystemExit("Check FAILED: the model could not memorise one batch")
```
- Take a **small real batch** (4 windows × 64 tokens; `seed + 1` for different windows), run the memorisation loop (Section 5.6), print first → last loss and the loss at five milestones.
- **Fail** if the final loss is still above 0.5: a correct model with a correct loss, gradients and optimizer **must** memorise a tiny batch. Failing means something is wired wrongly.

### 7.5 Report, log, exit

```python
    write_report(a.report, a, device_name, prec, vocab, train_tokens, reports, init_loss, attn_diff, losses)
    print(f"\nAll checks PASS. Report: {a.report}\nConfigs: {a.configs_dir}")
    if not a.no_log:
        append_run_log(a.run_log, a, device_name, prec, reports, init_loss, attn_diff, losses)
        print(f"Appended an entry to {a.run_log.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```
- Write the report, print PASS and locations (reached only if all checks passed), log unless `--no-log`, return 0. Same ending as always.

---

## 8. The whole run in order

1. Options → check Day 6/7 files → device and precision.
2. Load vocabulary, check it matches the token file, open `train.bin` as a memory map.
3. Take one real batch.
4. For tiny / small / medium: save the config, build, check the parameter count, **time a training step**, print.
5. Initial loss ≈ `ln(vocab)`? Manual = fast attention?
6. Memorise one real batch → loss near 0?
7. Report → run log.

## 9. How to read the results

- **Parameters:** about 1M / 5M / 15M for the three sizes (exact numbers for your vocabulary are in the table).
- **Training tokens per parameter:** if it is **small** (well below the often-quoted rough rule of thumb of about 20 tokens per parameter), the bigger model may **over-fit** (memorise) the training data; your corpus is also made from templates, so its real variety is lower than its size. Don't decide here — **Day 11 decides by validation loss.**
- **ms per step / tokens per second:** how long Day 9 will take. Step time × number of steps ≈ total training time (real training adds the optimizer and data loading).
- **Peak GPU memory:** must fit your card (RTX 2060 = 6 GB, RTX 4060 = 8 GB). If `medium` is too big, lower `--batch`.
- **Initial loss ≈ ln(vocab):** the model starts as a uniform guesser (good).
- **Attention max difference ~1e-6 or smaller:** the fast and manual attention agree.
- **Memorise one batch:** the loss should fall from about `ln(vocab)` to near 0.

## 10. Small things worth knowing

- **The timing has no optimizer step and no data loading**, so real training will be slower per step.
- **bf16 on an old GPU:** if the report shows `bf16` on an RTX 2060, PyTorch may be emulating it (see Section 5.2). Real bf16 needs compute capability 8+ (Day 1's `native_bf16`).
- **Same batch for all sizes** makes the comparison fair, but one batch is a small sample for timing; run-to-run noise of a few percent is normal.
- **Only the configs are saved here, not weights.** No model is trained in this script (the memorising model is thrown away).
- **Configs depend on the vocabulary.** If you retrain the tokenizer (Day 6), rerun Day 7 and Day 8 so the token files and configs match.
- **Checks run in order and stop at the first failure**; the report and configs of the sizes are saved only if everything passes (the configs for the three sizes are actually saved earlier, during step 4).

## 11. Quick summary in one paragraph

The script builds Model A at three sizes using the real vocabulary, counts the parameters (cross-checked with a formula), times one forward + backward step on real token windows to get speed and GPU memory, proves the untrained model starts at `ln(vocabulary)` loss, that the fast and manual attention agree, and that a model can memorise one real batch (so the model, loss, gradients and optimizer are wired correctly), and then saves the three configs for Day 9/11 and writes a report and a run-log entry.
