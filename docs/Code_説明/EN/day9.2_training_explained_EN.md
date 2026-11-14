# Day 9.2 — training.py — Simple Line-by-Line Explanation (English)

> **How the five Day 9 files fit together**
> - **Day 9.1 `batching.py`** — makes the batches (read it first).
> - **Day 9.2 (this file) `training.py`** — the **training loop**: the heart of Day 9 (also used on Day 11 and 12).
> - **Day 9 `day9_train.py`** — the command-line script that calls `train(...)`.
> - **Day 9 `day9_compare_runs.py`** — proves stop-and-resume works.
> - **Day 9.3 `test_training.py`** — automatic tests.
>
> This is the biggest file so far. Take sections 0 and 1 slowly; the rest follows them.

---

## 0. The big picture (read this first)

Training = repeating the same small cycle thousands of times:

```
take a batch of windows  →  model predicts the next tokens  →  measure the loss
→  compute gradients (which way to change each parameter)  →  change the parameters a little
```

One full cycle is called one **step**. This file wraps that cycle with everything a *serious* training run needs:

| Need | How `training.py` solves it |
|---|---|
| Learning rate that starts gently and ends gently | **Warm-up + cosine decay** (`lr_at`) |
| Bigger effective batch on a small GPU | **Gradient accumulation** (several micro-batches per step) |
| Keep training stable | **Gradient clipping**, **AdamW**, weight decay only on matrices |
| Fast and memory-friendly | **Mixed precision** (bf16, or fp16 + scaler) |
| Know how learning is going | **log.csv** every `log_every` steps; **validation loss** and **sample texts** every `eval_every` steps |
| Never lose work | **Checkpoints** `last.pt` and `best.pt`, saved safely |
| Continue after a stop | **Resume** with *exactly* the same batches and learning rates |
| Stop politely | **Ctrl+C** finishes the step and saves |
| Catch disasters | Stops and saves `diverged.pt` if the loss becomes NaN/infinity |

### One optimizer step (from the docstring)

1. learning rate for this step (warm-up, then cosine decay to `min_lr`);
2. `grad_accum` micro-batches: forward (autocast), loss, backward;
3. clip the gradient norm to `grad_clip`;
4. AdamW update (weight decay only on matrices, not on norm gains).

---

## 1. Words you need to know (mini dictionary)

| Word | Simple meaning |
|---|---|
| **Learning rate (lr)** | How big a change is made to the parameters at each step. Too big → unstable; too small → slow. |
| **Warm-up** | Start with a tiny learning rate and raise it linearly, so the first (random, chaotic) steps do not wreck the model. |
| **Cosine decay** | After warm-up, lower the learning rate smoothly along a cosine curve down to `min_lr`: big steps early, fine steps at the end. |
| **Gradient accumulation** | Run several small batches, **add up** their gradients, and update once. Effective batch = `batch_size × grad_accum`, without needing the memory for one huge batch. |
| **Gradient clipping** | If the gradient is huge (a "spike"), scale it down so its overall size is at most `grad_clip`. Prevents explosions. |
| **AdamW** | The optimizer: Adam (adapts the step size for each parameter) + decoupled weight decay. `beta1`, `beta2` control its memory (0.9 / 0.95). |
| **Weight decay** | A gentle pull of weights towards 0 to discourage over-fitting. Applied to **matrices only**, not to the norm gains (vectors). |
| **GradScaler** | The fp16 helper from Day 1 (multiply the loss by a big number so tiny gradients don't become 0). |
| **Perplexity** | `exp(loss)`. Roughly "among how many equally likely tokens the model is choosing". A uniform guess over V tokens has perplexity V; lower is better. |
| **Checkpoint** | A saved snapshot of everything needed to continue (model, optimizer, …). |
| **Resume** | Continue a stopped run from its checkpoint. |
| **RNG state** | The internal state of a random generator; restoring it makes the following "random" numbers repeat exactly. |
| **Greedy decoding** | Always pick the single highest-scoring next token. (Day 10 adds proper sampling.) |
| **Signal / SIGINT** | The message the operating system sends when you press **Ctrl+C**. |
| **Context manager (`with`)** | An object that does setup on entering a `with` block and cleanup on leaving it. |
| **Closure / `nonlocal`** | A function defined inside another function that can read (and with `nonlocal`, change) the outer function's variables. |
| **Decorator `@torch.no_grad()`** | Wraps a function so it runs without recording gradients (faster, uses less memory). |

---

## 2. The docstring, imports and constants

- The docstring (English + Japanese) describes the optimizer step, logging and evaluation, **resuming** (what `last.pt` holds), and **stopping safely** (Ctrl+C; checkpoints written to a temporary file then renamed).

```python
from __future__ import annotations

import csv, hashlib, json, math, os, platform, signal, time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import torch

from src.batching import RandomWindows, open_tokens, validation_batches, validation_windows
from src.model import ModelA, ModelConfig
from src.token_data import read_meta
```
- Most are known from earlier days. New: `os` (file renaming), `signal` (Ctrl+C handling), `fields` (list a dataclass's fields), `platform` (Python version).
- From the other files of the project: the batching helpers (**Day 9.1**), the model (**Day 8.1**) and `read_meta` (**Day 7.1**).

```python
CHECKPOINT_FORMAT: str = "model-a-checkpoint-v1"
LOG_COLUMNS: tuple[str, ...] = ("step", "event", "tokens_seen", "lr", "train_loss", "train_ppl", "val_loss",
                                "val_ppl", "grad_norm", "tokens_per_sec", "peak_vram_mb", "elapsed_s")
DEFAULT_PROMPTS: tuple[str, ...] = ("営業部の", "ボーナスが", "大阪支社で", "社員の平均")
RESUME_FREE_KEYS: frozenset[str] = frozenset({"stop_after", "resume", "device", "machine", "overwrite"})
```
- `CHECKPOINT_FORMAT` — a name tag in every checkpoint (same idea as Day 6.1/7.1).
- `LOG_COLUMNS` — the 12 columns of `log.csv`.
- `DEFAULT_PROMPTS` — the fixed text beginnings used for sample outputs ("営業部の", "ボーナスが", …).
- `RESUME_FREE_KEYS` — the settings that **may differ** when resuming, because they do not change the batches or the learning curve (when to stop, whether to resume, which device, labels).

---

## 3. `TrainConfig` — all the settings

```python
@dataclass
class TrainConfig:
    run_dir: str
    model_config: str = "configs/model_tiny.json"
    tokens_dir: str = "data/tokens"
    tokenizer: str | None = "data/tokenizer/tokenizer.json"
    batch_size: int = 32
    grad_accum: int = 1
    max_steps: int = 600
    warmup_steps: int = 60
    lr: float = 1e-3
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    dropout: float | None = None
    log_every: int = 10
    eval_every: int = 100
    eval_max_windows: int | None = None
    sample_tokens: int = 40
    prompts: list[str] = field(default_factory=lambda: list(DEFAULT_PROMPTS))
    precision: str = "auto"
    device: str = "auto"
    seed: int = 42
    stop_after: int | None = None
    resume: bool = False
    overwrite: bool = False
    machine: str = "home"
```

| Setting | Meaning |
|---|---|
| `run_dir` | Folder of this run (required, e.g. `runs/day9/smoke_a`) |
| `model_config` | Which `configs/model_*.json` (Day 8) to use |
| `tokens_dir`, `tokenizer` | Where the Day 7 token files and the Day 6 tokenizer are (`tokenizer=None` skips the tokenizer check and sample texts) |
| `batch_size`, `grad_accum` | Windows per micro-batch, and micro-batches per step |
| `max_steps` | Total optimizer steps |
| `warmup_steps` | Steps of linear warm-up |
| `lr`, `min_lr_ratio` | Peak learning rate and the final rate as a fraction of it (0.1 → ends at 10%) |
| `weight_decay` | Strength of weight decay (matrices only) |
| `beta1`, `beta2` | AdamW's memory constants (0.9, 0.95) |
| `grad_clip` | Maximum gradient norm |
| `dropout` | `None` = keep the model config's dropout; a number overrides it |
| `log_every`, `eval_every` | How often to log / evaluate (steps) |
| `eval_max_windows` | Limit validation windows for speed (`None` = all) |
| `sample_tokens` | Length of each sample continuation (40 tokens) |
| `prompts` | The fixed text beginnings (`default_factory=lambda: ...` gives each object its own list) |
| `precision` | `auto` / `bf16` / `fp16` / `fp32` |
| `device` | `auto` / `cuda` / `cpu` |
| `seed` | Random seed (model initialisation and the batch generator) |
| `stop_after` | Stop at this step on purpose (simulates an interruption) |
| `resume`, `overwrite` | Resume a run / start again in an existing folder |
| `machine` | Label for logs |

```python
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TrainConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})
```
- `to_dict` — settings as a plain dictionary (saved as `train_config.json`).
- `from_dict` — rebuild from a dictionary, **ignoring unknown keys** (`fields(cls)` lists the real field names). So an older `train_config.json` still loads even if a setting was added or removed later.

---

## 4. Small pieces

### 4.1 `lr_at` — the learning-rate schedule

```python
def lr_at(step: int, max_lr: float, warmup: int, max_steps: int, min_ratio: float) -> float:
    """Learning rate for the step about to be taken (step counts from 0)."""
    min_lr = max_lr * min_ratio
    if step < warmup:
        return max_lr * (step + 1) / warmup
    if step >= max_steps:
        return min_lr
    progress = (step - warmup) / max(1, max_steps - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
```
- `min_lr` = the final learning rate.
- **Warm-up** (`step < warmup`): `max_lr × (step+1) / warmup` — rises linearly from `max_lr/warmup` to `max_lr`.
- After the end (`step >= max_steps`): `min_lr`.
- **Cosine decay**: `progress` goes from 0 to 1 over the remaining steps (`max(1, …)` avoids dividing by zero). The formula `min_lr + 0.5 × (max_lr − min_lr) × (1 + cos(π × progress))` starts at `max_lr` (progress 0: cos 0 = 1) and ends at `min_lr` (progress 1: cos π = −1), with a smooth S-shaped drop in between.
- Example (max 1e-3, warm-up 10, 100 steps): step 0 → 1e-4; step 9 → 1e-3; step 55 (halfway) → 5.5e-4; step 100 → 1e-4. (Day 9.3 tests exactly these.)

### 4.2 `param_groups` — weight decay only on matrices

```python
def param_groups(model: torch.nn.Module, weight_decay: float) -> list[dict]:
    decay = [p for _, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for _, p in model.named_parameters() if p.requires_grad and p.dim() < 2]
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
```
- Splits the parameters into two groups for the optimizer:
  - **matrices** (`p.dim() >= 2`: the embedding, attention and MLP weights) → get `weight_decay`;
  - **vectors** (`p.dim() < 2`: the RMSNorm gains) → **no** decay (shrinking a gain toward 0 would just hurt).
- `named_parameters()` lists each shared (tied) tensor once. `requires_grad` = "is learnable".

### 4.3 `pick_device`, `pick_precision`, `autocast_ctx`, `sha256_file`

```python
def pick_device(choice: str) -> torch.device: ...
def pick_precision(choice: str, device: torch.device) -> str:
    if device.type != "cuda":
        return "fp32"
    if choice == "auto":
        return "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    if choice == "bf16" and not torch.cuda.is_bf16_supported():
        raise SystemExit("this GPU does not support bf16; use --precision fp16")
    return choice

def autocast_ctx(device: torch.device, precision: str):
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(precision)
    return torch.autocast(device_type=device.type, dtype=dtype) if dtype is not None else nullcontext()
```
- `pick_device` — same as Day 8.
- `pick_precision` — CPU → fp32; `auto` → bf16 if supported else fp16; if you **force** bf16 on a GPU that lacks it → stop with a hint. (Same caution as Day 8: on some PyTorch versions `is_bf16_supported()` may say yes for an old GPU by emulation; the printed precision tells you what was chosen.)
- `autocast_ctx` — same idea as Day 8's `autocast`: a dictionary maps the precision name to a dtype (`.get` returns `None` for fp32 → `nullcontext()`).
- `sha256_file` — same as Day 4.

### 4.4 `evaluate`

```python
@torch.no_grad()
def evaluate(model, tokens, block_size, batch_size, device, precision, max_windows) -> float:
    """Average next-token loss over the fixed validation windows."""
    model.eval()
    total, count = 0.0, 0
    for x, y in validation_batches(tokens, block_size, batch_size, max_windows):
        x, y = x.to(device), y.to(device)
        with autocast_ctx(device, precision):
            _, loss = model(x, y)
        total += loss.item() * x.size(0)
        count += x.size(0)
    model.train()
    return total / max(1, count)
```
- `@torch.no_grad()` — no gradients inside (decorator).
- Evaluation mode; for every fixed validation batch: move to the device, forward pass, get the batch's **mean loss**, multiply by the number of windows in the batch (`x.size(0)`) and add to `total`. Dividing by the total number of windows gives the **exact average** even if the last batch is smaller.
- Back to training mode; return the average loss (`max(1, count)` avoids dividing by zero).

### 4.5 `greedy_continue`

```python
@torch.no_grad()
def greedy_continue(model, ids, max_new, eos_id, device) -> list[int]:
    """Always take the highest-scoring next token (Day 10 adds proper sampling)."""
    model.eval()
    out = list(ids)
    for _ in range(max_new):
        x = torch.tensor([out[-model.cfg.block_size:]], dtype=torch.long, device=device)
        logits, _ = model(x)
        nxt = int(logits[0, -1].argmax())
        out.append(nxt)
        if nxt == eos_id:
            break
    model.train()
    return out[len(ids):]
```
- Writes a continuation of a prompt, one token at a time:
  - keep the last `block_size` tokens (`out[-block_size:]`) as input (a batch of one);
  - `logits[0, -1]` = the scores at the **last position**; `.argmax()` = the **highest-scoring token**; add it;
  - stop at the `<eos>` token or after `max_new` tokens.
- Returns only the **new** tokens (`out[len(ids):]`). "Greedy" is deterministic, so the sample texts of different runs are comparable.

### 4.6 Checkpoints: `save_checkpoint`, `load_checkpoint`

```python
def save_checkpoint(path: Path, state: dict) -> None:
    """Write to a temporary file, then rename: an interruption never leaves a half-written file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)

def load_checkpoint(path: Path, device: torch.device) -> dict:
    state = torch.load(path, map_location=device, weights_only=True)
    if state.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path}: not a Model A checkpoint")
    return state
```
- `save_checkpoint`: write to `last.pt.tmp` first, then `os.replace(tmp, path)` swaps it into place in one **atomic** step. If the power fails mid-write, the old `last.pt` is still intact.
- `load_checkpoint`: `torch.load(..., weights_only=True)` loads **only numbers/tensors and simple containers** (it refuses to run arbitrary code hidden in a file — a safety feature); `map_location=device` puts tensors on the right device. Check the format tag.

### 4.7 `_StopFlag` — polite Ctrl+C

```python
class _StopFlag:
    """Ctrl+C once = stop after the current step and save; twice = stop immediately."""

    def __init__(self) -> None:
        self.requested = False
        self._old = None

    def __enter__(self) -> "_StopFlag":
        try:
            self._old = signal.signal(signal.SIGINT, self._handler)
        except ValueError:          # not in the main thread (for example inside some test runners)
            self._old = None
        return self

    def _handler(self, signum, frame) -> None:  # noqa: ARG002
        if self.requested:
            raise KeyboardInterrupt
        self.requested = True
        print("\nCtrl+C: finishing this step, saving last.pt, then stopping (press again to stop at once).")

    def __exit__(self, *exc) -> None:
        if self._old is not None:
            signal.signal(signal.SIGINT, self._old)
```
- A **context manager** used as `with _StopFlag() as stop:`.
- `__enter__` (runs when the `with` block starts): `signal.signal(SIGINT, handler)` installs **our own Ctrl+C handler** and remembers the old one. (If we are not in the main thread, Python raises `ValueError`; then nothing is installed.)
- `_handler` (runs when Ctrl+C is pressed): the **first** press only sets `requested = True` and prints a message — the training loop checks this flag after the current step. The **second** press raises `KeyboardInterrupt` to stop at once.
- `__exit__` (runs when the block ends): **restore** the previous handler.

### 4.8 `_truncate_after` — cleaning up after a hard kill

```python
def _truncate_after(log_path: Path, samples_path: Path, step: int) -> None:
    if log_path.exists():
        with log_path.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        keep = [r for r in rows if int(r["step"]) <= step]
        if len(keep) != len(rows):
            with log_path.open("w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
                w.writeheader()
                w.writerows(keep)
            print(f"Removed {len(rows) - len(keep)} log rows after step {step} (they will be trained again)")
    if samples_path.exists():
        text = samples_path.read_text(encoding="utf-8")
        parts = text.split("\n## step ")
        kept = [parts[0]] + [p for p in parts[1:] if int(p.split(" ", 1)[0]) <= step]
        if len(kept) != len(parts):
            samples_path.write_text("\n## step ".join(kept), encoding="utf-8", newline="\n")
```
- **The problem:** if the program is *killed* (window closed, power cut) rather than stopped with Ctrl+C, `log.csv` and `samples.md` may contain steps **after** the last saved checkpoint. Those steps will be trained again on resume, so their old lines must go, otherwise the log would have duplicates.
- `log.csv`: read all rows, keep those with `step <= checkpoint step`; if anything was removed, rewrite the file (header + kept rows) and print how many were removed.
- `samples.md`: split the text into sections at `"\n## step "`; `parts[0]` is the header; each other part starts with its step number (`p.split(" ", 1)[0]`); keep sections up to the checkpoint step and rewrite the file if needed.

### 4.9 `_fmt`

```python
def _fmt(v: float | None, digits: int = 4) -> str:
    return "" if v is None else f"{v:.{digits}f}"
```
- Format a number with a given number of decimals, or an empty string for `None` (the nested `{v:.{digits}f}` takes the number of decimals from the variable `digits`).

---

## 5. `train` — the whole run

```python
def train(cfg: TrainConfig, repo_root: Path) -> dict:
    """Train (or resume) one run. Returns a short summary."""
```
Takes the settings and the repo folder; returns a summary dictionary.

### 5.1 Paths, device and precision

```python
    root = repo_root
    run_dir = (root / cfg.run_dir).resolve()
    tokens_dir = (root / cfg.tokens_dir).resolve()
    device = pick_device(cfg.device)
    precision = pick_precision(cfg.precision, device)
```
- Absolute paths for the run folder and token files; choose device and precision.

### 5.2 Check settings and data against each other

```python
    mcfg = ModelConfig.load((root / cfg.model_config).resolve())
    if cfg.dropout is not None:
        mcfg.dropout = cfg.dropout
    train_meta = read_meta(tokens_dir, "train")
    if train_meta.vocab_size != mcfg.vocab_size:
        raise SystemExit("model config and token files use different vocabulary sizes")
```
- Load the model config (Day 8); optionally override dropout; read the training meta label (Day 7.1). **The vocabulary sizes must match** (same check as Day 8).

```python
    tok = None
    if cfg.tokenizer:
        from src.bpe_tokenizer import ByteBPETokenizer
        tok_path = (root / cfg.tokenizer).resolve()
        if sha256_file(tok_path) != train_meta.tokenizer_sha256:
            raise SystemExit("data/tokens was made with a different tokenizer; rerun Day 7")
        tok = ByteBPETokenizer.load(tok_path)
    train_tokens = open_tokens(tokens_dir, "train")
    val_tokens = open_tokens(tokens_dir, "val")
```
- If a tokenizer is given: compute its **SHA-256** and compare with the fingerprint stored in the token file's meta (`tokenizer_sha256`). A mismatch means the tokens were made with a different tokenizer → stop. (This is the *stronger* check that Day 7.1's notes said the trainer should do; Day 8 only compared the vocabulary size.) Load the tokenizer (needed for sample texts). The import is inside `if` so the tokenizer module is only needed when used.
- Open the train and validation token streams (Day 9.1).

### 5.3 The run folder: new run or resume

```python
    last_path, best_path = run_dir / "last.pt", run_dir / "best.pt"
    log_path, samples_path = run_dir / "log.csv", run_dir / "samples.md"
    if cfg.resume:
        if not last_path.exists():
            raise SystemExit(f"nothing to resume: {last_path} does not exist")
        saved = json.loads((run_dir / "train_config.json").read_text(encoding="utf-8"))
        diff = [k for k, v in cfg.to_dict().items() if k not in RESUME_FREE_KEYS and saved.get(k) != v]
        if diff:
            raise SystemExit(f"resume refused: settings differ from the first run: {diff}")
    else:
        if run_dir.exists() and any(run_dir.iterdir()) and not cfg.overwrite:
            raise SystemExit(f"{run_dir} already has files. Use a new --run-name, --resume, or --overwrite.")
        run_dir.mkdir(parents=True, exist_ok=True)
        for p in (last_path, best_path, log_path, samples_path):
            p.unlink(missing_ok=True)
        (run_dir / "train_config.json").write_text(json.dumps(cfg.to_dict(), indent=2, ensure_ascii=False),
                                                   encoding="utf-8", newline="\n")
        mcfg.save(run_dir / "model_config.json")
```
- **Resume:** `last.pt` must exist. Compare the current settings with the saved `train_config.json`: any difference **except the "free" keys** → **refuse** (resuming with a different learning rate would silently produce a different run).
- **New run:** refuse if the folder already contains files (unless `--overwrite`) — protects old runs from being overwritten. Create the folder, delete leftover old files, and **save the settings** (`train_config.json`) and the model settings (`model_config.json`) — so a run is always reproducible.

### 5.4 Build the model, optimizer, scaler and batch generator

```python
    torch.manual_seed(cfg.seed)
    model = ModelA(mcfg).to(device)
    use_fused = device.type == "cuda"
    opt = torch.optim.AdamW(param_groups(model, cfg.weight_decay), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2),
                            fused=use_fused)
    scaler = torch.amp.GradScaler(device.type, enabled=(precision == "fp16"))
    windows = RandomWindows(train_tokens, mcfg.block_size, cfg.batch_size, cfg.seed)
```
- Seed PyTorch's random generator, build the model on the device.
- **AdamW** with the two parameter groups, learning rate, betas; `fused=True` on a GPU = a faster implementation.
- **GradScaler** is **enabled only for fp16** (for bf16/fp32 it does nothing). *Note:* this uses the newer `torch.amp.GradScaler(device_type, …)` form; Day 1's smoke test had a fallback for older PyTorch, this one does not.
- The batch generator (Day 9.1) with the same seed.

### 5.5 Initial counters and restoring from a checkpoint

```python
    step, best_val, elapsed_before, tokens_seen = 0, float("inf"), 0.0, 0
    if cfg.resume:
        st = load_checkpoint(last_path, device)
        if st["model_config"] != mcfg.to_dict():
            raise SystemExit("resume refused: the model settings differ from the checkpoint")
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["optimizer"])
        scaler.load_state_dict(st["scaler"])
        windows.load_state_dict(st["windows"])
        torch.set_rng_state(st["torch_rng"].cpu())
        if device.type == "cuda" and st.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all([s.cpu() for s in st["cuda_rng"]])
        step, best_val = st["step"], st["best_val"]
        elapsed_before, tokens_seen = st["elapsed_s"], st["tokens_seen"]
        _truncate_after(log_path, samples_path, step)
        print(f"Resumed from {last_path.name} at step {step} (best validation loss so far {best_val:.4f})")
```
- Counters: `step` = completed steps, `best_val` = lowest validation loss so far (starts at infinity), `elapsed_before` = time already spent, `tokens_seen`.
- On **resume**, restore **everything**:
  - check that the model settings match the checkpoint;
  - model weights, **optimizer state** (Adam's running averages!), scaler state, the **batch generator's state** (Day 9.1), PyTorch's CPU and CUDA random states;
  - `step`, `best_val`, `elapsed`, `tokens_seen`;
  - clean log/samples after that step (Section 4.8).
- This is why a resumed run continues with **exactly the same batches and learning rates**, and the loss curve does not jump.

### 5.6 Announce the setup

```python
    tokens_per_step = cfg.batch_size * cfg.grad_accum * mcfg.block_size
    n_val = validation_windows(val_tokens, mcfg.block_size)
    if cfg.eval_max_windows is not None:
        n_val = min(n_val, cfg.eval_max_windows)
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print(...)  # device | precision | model | parameter count;  batch x accum x block = tokens/step | steps | validation windows
```
- `tokens_per_step` = tokens processed per optimizer step. Count validation windows. Print a two-line summary.

### 5.7 The closures: `state_for_save`, `write_row`, `checkpoint_and_sample`

```python
    def state_for_save() -> dict:
        return {
            "format": CHECKPOINT_FORMAT, "step": step, "model": model.state_dict(), "optimizer": opt.state_dict(),
            "scaler": scaler.state_dict(), "windows": windows.state_dict(), "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
            "model_config": mcfg.to_dict(), "train_config": cfg.to_dict(), "best_val": best_val,
            "elapsed_s": elapsed_before + (time.perf_counter() - t_start), "tokens_seen": tokens_seen,
            "tokenizer_sha256": train_meta.tokenizer_sha256, "torch_version": str(torch.__version__),
            "python_version": platform.python_version(),
        }
```
- What goes **into a checkpoint**: the format tag, current step, model weights, optimizer, scaler, the batch generator state, PyTorch's random states (CPU and GPU), both configs, `best_val`, elapsed time, tokens seen, the tokenizer fingerprint, and the PyTorch/Python versions (for the record). Being defined **inside** `train`, it reads the *current* values of `step`, `best_val`, … at the moment it is called (a closure).

```python
    new_log = not log_path.exists()
    log_f = log_path.open("a", encoding="utf-8", newline="")
    writer = csv.DictWriter(log_f, fieldnames=LOG_COLUMNS)
    if new_log:
        writer.writeheader()
```
- Open `log.csv` in **append** mode (so a resumed run adds to it); write the header only for a new file.

```python
    def write_row(event, lr, train_loss, val_loss, grad_norm, tps) -> None:
        peak = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None
        writer.writerow({
            "step": step, "event": event, "tokens_seen": tokens_seen, "lr": f"{lr:.6g}",
            "train_loss": _fmt(train_loss), "train_ppl": _fmt(math.exp(train_loss) if train_loss is not None else None, 2),
            "val_loss": _fmt(val_loss), "val_ppl": _fmt(math.exp(val_loss) if val_loss is not None else None, 2),
            "grad_norm": _fmt(grad_norm), "tokens_per_sec": _fmt(tps, 0), "peak_vram_mb": _fmt(peak, 0),
            "elapsed_s": f"{elapsed_before + time.perf_counter() - t_start:.1f}",
        })
        log_f.flush()
```
- Writes **one row** of `log.csv`. `event` is `"train"` or `"eval"`. Perplexity = `exp(loss)`. Missing values become empty strings. `log_f.flush()` forces the row onto the disk immediately, so a crash loses nothing.

```python
    def checkpoint_and_sample(lr: float) -> float:
        nonlocal best_val
        val = evaluate(model, val_tokens, mcfg.block_size, cfg.batch_size, device, precision, cfg.eval_max_windows)
        improved = val < best_val
        best_val = min(best_val, val)
        write_row("eval", lr, None, val, None, None)
        if tok is not None and cfg.prompts:
            lines = [f"\n## step {step} (validation loss {val:.4f})\n", "| Prompt | Greedy continuation |", "| --- | --- |"]
            for p in cfg.prompts:
                cont = greedy_continue(model, [1] + tok.encode(p), cfg.sample_tokens, 2, device)
                text = tok.decode(cont).replace("|", "\\|").replace("\n", " ")
                lines.append(f"| {p} | {text}{' ⟨eos⟩' if cont and cont[-1] == 2 else ''} |")
            with samples_path.open("a", encoding="utf-8", newline="\n") as f:
                f.write("\n".join(lines) + "\n")
        state = state_for_save()
        save_checkpoint(last_path, state)
        if improved:
            save_checkpoint(best_path, state)
        print(f"step {step:>6} | validation loss {val:.4f} (perplexity {math.exp(val):.2f})"
              f"{'  * best, saved best.pt' if improved else ''} | saved last.pt")
        return val
```
- **The "evaluation moment":**
  1. compute the **validation loss** (Section 4.4); `improved` = lower than ever before; update `best_val` (`nonlocal` lets this inner function change the outer variable);
  2. write an `eval` row;
  3. if there is a tokenizer: for every fixed prompt, build the input `[1] + tok.encode(p)` (**`1` = `<bos>`**), generate a greedy continuation (stop at `2` = `<eos>`), decode to text, **escape `|` and line breaks** so the Markdown table does not break, and append a section `## step N (validation loss …)` to `samples.md`. You can watch the text improve over time.
  4. save `last.pt` (always) and `best.pt` (only when validation improved);
  5. print a line.

### 5.8 The main loop

```python
    t_start = time.perf_counter()
    model.train()
    interval_loss = torch.zeros((), device=device)
    interval_steps, interval_t0 = 0, time.perf_counter()
    stopped_early = False
    last_lr = cfg.lr
    try:
        with _StopFlag() as stop:
            if step == 0 and not cfg.resume:
                checkpoint_and_sample(lr_at(0, cfg.lr, cfg.warmup_steps, cfg.max_steps, cfg.min_lr_ratio))
                interval_t0 = time.perf_counter()
            while step < cfg.max_steps:
```
- Start the stopwatch, training mode. `interval_loss` — a GPU tensor that **accumulates the losses of the steps since the last log** (so we only copy to the CPU when logging; copying every step would slow the GPU).
- Install the Ctrl+C handler (`with _StopFlag() as stop`).
- **Before any training** (step 0 of a new run): do one evaluation → the **baseline** validation loss (≈ `ln(vocab)`) and the first sample texts, plus the first checkpoint.
- Loop until `max_steps` steps are done.

```python
                lr = lr_at(step, cfg.lr, cfg.warmup_steps, cfg.max_steps, cfg.min_lr_ratio)
                last_lr = lr
                for g in opt.param_groups:
                    g["lr"] = lr
```
- This step's learning rate; set it in every optimizer group.

```python
                step_loss = torch.zeros((), device=device)
                for _ in range(cfg.grad_accum):
                    x, y = windows.next_batch()
                    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                    with autocast_ctx(device, precision):
                        _, loss = model(x, y)
                    step_loss += loss.detach() / cfg.grad_accum
                    scaler.scale(loss / cfg.grad_accum).backward()
```
- **Gradient accumulation loop** (`grad_accum` micro-batches; 1 by default):
  - get a batch (Day 9.1) and move it to the device (`non_blocking=True` allows overlapping copy and compute);
  - forward pass under autocast → loss;
  - `step_loss += loss.detach() / grad_accum` — keep the **average loss** of this step for logging (`detach()` = no gradient tracking);
  - `scaler.scale(loss / grad_accum).backward()` — backward pass. Dividing by `grad_accum` makes the **sum of gradients equal the average** over the micro-batches. (`scaler.scale` multiplies by the fp16 scaling factor; with bf16/fp32 it does nothing.) Gradients **add up** inside the parameters across the micro-batches.

```python
                scaler.unscale_(opt)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                step += 1
                tokens_seen += tokens_per_step
                interval_loss += step_loss
                interval_steps += 1
```
- `scaler.unscale_(opt)` — undo the fp16 scaling so the gradients have their true size.
- `clip_grad_norm_(…, grad_clip)` — if the total gradient size exceeds `grad_clip`, **scale it down**; it returns the size **before** clipping (`grad_norm`, logged).
- `scaler.step(opt)` — the **AdamW update** (the scaler skips it if it finds infinities); `scaler.update()` adjusts the scale; `zero_grad` clears the gradients.
- Count the step: `step += 1`, `tokens_seen`, and add to the interval sums.

```python
                if step % cfg.log_every == 0 or step == cfg.max_steps:
                    mean_loss = (interval_loss / interval_steps).item()
                    if not math.isfinite(mean_loss):
                        save_checkpoint(run_dir / "diverged.pt", state_for_save())
                        raise SystemExit(f"loss became {mean_loss} at step {step}; saved diverged.pt and stopped")
                    dt = time.perf_counter() - interval_t0
                    tps = interval_steps * tokens_per_step / dt if dt > 0 else None
                    write_row("train", lr, mean_loss, None, float(grad_norm), tps)
                    print(f"step {step:>6} | loss {mean_loss:.4f} | lr {lr:.2e} | {(tps or 0):,.0f} tokens/s")
                    interval_loss.zero_()
                    interval_steps, interval_t0 = 0, time.perf_counter()
```
- **Logging** every `log_every` steps (and at the last step): `mean_loss` = average loss over the steps since the last log (`.item()` = the one GPU→CPU copy). **If the loss is NaN or infinity**, save `diverged.pt` (for investigation) and **stop**. Otherwise compute tokens per second, write a `train` row, print a line, and reset the interval counters.

```python
                stop_now = stop.requested or (cfg.stop_after is not None and step >= cfg.stop_after)
                if stop_now and interval_steps > 0:      # log the unfinished interval before stopping
                    mean_loss = (interval_loss / interval_steps).item()
                    write_row("train", lr, mean_loss, None, float(grad_norm), None)
                    interval_loss.zero_()
                    interval_steps = 0
                if step % cfg.eval_every == 0 or step == cfg.max_steps:
                    checkpoint_and_sample(lr)
                    interval_t0 = time.perf_counter()
                elif stop_now:
                    save_checkpoint(last_path, state_for_save())
                    print(f"step {step:>6} | saved last.pt")
                if stop_now and step < cfg.max_steps:
                    stopped_early = True
                    print(f"Stopped at step {step}. Continue with --resume.")
                    break
    finally:
        log_f.close()
```
- `stop_now` — Ctrl+C was pressed, **or** `stop_after` reached (the deliberate "interruption" for the Day 9 test).
- If stopping with an **unfinished log interval**, write its average first (so no steps go unlogged).
- **Evaluation** every `eval_every` steps (and at the end) → `checkpoint_and_sample` (which also saves `last.pt`). If we are stopping at a step that is **not** an evaluation step, just save `last.pt`.
- If stopping early: mark `stopped_early`, print how to resume, and `break` out of the loop.
- `finally: log_f.close()` — **always** close the log file, even after an error.

### 5.9 The summary

```python
    return {"run_dir": str(run_dir), "step": step, "max_steps": cfg.max_steps, "best_val": best_val,
            "stopped_early": stopped_early, "device": device_name, "precision": precision, "last_lr": last_lr,
            "elapsed_s": elapsed_before + time.perf_counter() - t_start}
```
- A small dictionary for the caller (`day9_train.py` prints it and writes the run log).

---

## 6. The run folder after a run

| File | Content |
|---|---|
| `train_config.json`, `model_config.json` | The settings (a resumed run must match them) |
| `log.csv` | One row per log/eval step (12 columns) |
| `samples.md` | Greedy continuations of fixed prompts at every evaluation |
| `last.pt` | Latest checkpoint (for resuming) |
| `best.pt` | Checkpoint with the lowest validation loss (for generation / final model) |
| `diverged.pt` | Only if the loss blew up |

## 7. Small things worth knowing

- **Stop at a multiple of `log_every` for an exact comparison.** If you stop at a step that is *not* a multiple of `log_every`, an extra partial-average row is written at the stop step and the first interval after resuming averages fewer steps, so a few logged `train_loss` values will differ slightly from an uninterrupted run (the actual training is still identical). The Day 9 smoke test stops at a multiple (300 with log-every 10).
- **Validation loss at step 0** is the baseline before any learning (should be ≈ `ln(vocab)` — Day 8's check).
- **`best.pt` vs `last.pt`:** use `best.pt` for generation; use `last.pt` to continue training.
- **The loss check happens only at log time**, so a NaN may take up to `log_every` steps to be noticed.
- **Weight decay applies to the embedding too** (it is a matrix), but not to the norm gains.
- **Perplexity** `exp(loss)` is easier to feel than loss: at the start it is about the vocabulary size; when learning works it falls steeply.
- **Only the main thread can catch Ctrl+C** with this handler (inside some test runners the handler is skipped).
- **Needs a PyTorch with `torch.amp.GradScaler(device_type, …)`** (see 5.4).

## 8. Quick summary in one paragraph

`training.py` runs the whole training: for each step it sets a warm-up-then-cosine learning rate, accumulates gradients over micro-batches in mixed precision, clips the gradient and updates with AdamW (weight decay on matrices only); it logs to `log.csv`, evaluates on fixed validation windows, writes greedy sample texts, and saves `last.pt` and `best.pt` safely (temporary file then rename); a checkpoint contains the model, optimizer, scaler, step and all random states including the batch generator's, so a resumed run continues exactly as an uninterrupted one, Ctrl+C stops politely, and a hard kill is cleaned up on resume.
