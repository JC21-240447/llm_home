# Day 9.3 — test_training.py — Simple Line-by-Line Explanation (English)

> **How the five Day 9 files fit together**
> - **Day 9.1 `batching.py`** and **Day 9.2 `training.py`** — the code being tested (read them first).
> - **Day 9 `day9_train.py`** and **`day9_compare_runs.py`** — the scripts.
> - **Day 9.3 (this file) `test_training.py`** — **8 automatic tests**, all on the **CPU in a few seconds**. The most important one: *a stopped-and-resumed run gives exactly the same losses as an uninterrupted run* (the same proof as `day9_compare_runs.py`, but automatic).
>
> pytest basics (`assert`, `parametrize`, `tmp_path`, `pytest.raises`, fixtures) are in Days 6.2, 7.2 and 8.2; only new things are explained here.

---

## 0. The big picture (read this first)

| Promise | Test |
|---|---|
| Learning rate rises linearly (warm-up), then falls like a cosine and never rises again | `test_lr_schedule_warmup_then_cosine` |
| Weight decay only on matrices; every parameter in exactly one group | `test_weight_decay_only_on_matrices` |
| `y` is `x` shifted by one; a saved batch-generator state reproduces the next batch | `test_random_windows_shift_and_resume_state` |
| Validation windows cover the stream with no gaps and no overlap | `test_validation_windows_cover_the_stream_without_overlap` |
| **Stopped + resumed = uninterrupted** | `test_stopped_and_resumed_run_equals_uninterrupted_run` |
| Resuming with changed settings is **refused** | `test_resume_refuses_changed_settings` |
| A new run in an existing folder is **refused** | `test_new_run_refuses_an_existing_folder` |
| After a **hard kill**, resuming removes the stale log lines | `test_resume_after_a_hard_kill_removes_lines_written_after_the_checkpoint` |

Run (as before): `.venv\Scripts\python.exe -m pytest tests -q`

---

## 1. Setup

```python
from __future__ import annotations

import csv, json, math, sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.batching import RandomWindows, validation_batches, validation_windows  # noqa: E402
from src.model import ModelA, ModelConfig  # noqa: E402
from src.token_data import FORMAT_VERSION, TokenFileMeta, encode_lines, sha256_bytes, write_token_files  # noqa: E402
from src.training import TrainConfig, lr_at, param_groups, train  # noqa: E402

VOCAB = 40
```
- Same setup as the earlier test files. It imports from **Day 9.1** (batching), **Day 8.1** (model), **Day 7.1** (token files, to build fake data) and **Day 9.2** (training). `VOCAB = 40` — a pretend vocabulary.

---

## 2. Tests of the small pieces

### 2.1 `test_lr_schedule_warmup_then_cosine`

```python
def test_lr_schedule_warmup_then_cosine() -> None:
    lrs = [lr_at(s, 1e-3, 10, 100, 0.1) for s in range(101)]
    assert lrs[0] == pytest.approx(1e-4)          # first warm-up step
    assert lrs[9] == pytest.approx(1e-3)          # end of warm-up = peak
    assert all(a >= b for a, b in zip(lrs[9:], lrs[10:]))  # then never rises
    assert lrs[100] == pytest.approx(1e-4)        # ends at min_lr
    assert lrs[55] == pytest.approx(1e-4 + 0.5 * 9e-4, rel=1e-6)  # halfway through the cosine
```
- Compute the learning rate for steps 0–100 with peak `1e-3`, warm-up 10, 100 steps, final ratio 0.1.
- Step 0 = `1e-4` (first warm-up step = peak ÷ 10); step 9 = `1e-3` (end of warm-up = peak).
- `all(a >= b for a, b in zip(lrs[9:], lrs[10:]))` — comparing each value with the **next** one: never rising after the peak.
- Step 100 = `1e-4` (the minimum). Step 55 (halfway through the cosine) = exactly the midpoint `1e-4 + 0.5 × 9e-4`. (`pytest.approx(…, rel=1e-6)` = "equal within 0.0001%".)

### 2.2 `test_weight_decay_only_on_matrices`

```python
def test_weight_decay_only_on_matrices() -> None:
    m = ModelA(ModelConfig(vocab_size=VOCAB, block_size=16, n_layer=2, n_head=2, d_model=32))
    decay, no_decay = param_groups(m, 0.1)
    assert all(p.dim() >= 2 for p in decay["params"])
    assert all(p.dim() == 1 for p in no_decay["params"])
    ids = [id(p) for g in (decay, no_decay) for p in g["params"]]
    assert len(ids) == len(set(ids)) == len(list(m.parameters()))   # every parameter exactly once (tied once)
```
- Build a tiny model; `param_groups` returns two group dictionaries (unpacked into `decay`, `no_decay`).
- Everything in `decay` is a **matrix** (≥ 2 dimensions); everything in `no_decay` is a **vector** (1 dimension — the norm gains).
- `ids` — the identity of every parameter in both groups. The chained check requires: **no duplicates** (`len(ids) == len(set(ids))`) and **the total equals the model's parameter count** — so every parameter is in exactly one group (the tied embedding counted once).

### 2.3 `test_random_windows_shift_and_resume_state`

```python
def test_random_windows_shift_and_resume_state() -> None:
    tokens = np.arange(1000, dtype="<u2")
    w = RandomWindows(tokens, block_size=16, batch_size=4, seed=0)
    x, y = w.next_batch()
    assert x.shape == y.shape == (4, 16)
    assert torch.equal(x[:, 1:], y[:, :-1])       # y is x shifted by one
    state = w.state_dict()
    expected = w.next_batch()
    w2 = RandomWindows(tokens, block_size=16, batch_size=4, seed=999)
    w2.load_state_dict(state)
    got = w2.next_batch()
    assert torch.equal(expected[0], got[0]) and torch.equal(expected[1], got[1])
```
- A fake stream `0, 1, 2, …, 999` makes the result easy to see.
- Shapes `(4, 16)`; and `x[:, 1:] == y[:, :-1]` — **`y` is `x` shifted by one** (the heart of next-token learning).
- Save the generator state (`state`), take the **next** batch (`expected`). Make a **second generator with a different seed** (999) and load the saved state into it; its next batch (`got`) must be **exactly the same** — proving that saving and restoring the state reproduces the future batches (what resume relies on). `torch.equal` = identical tensors.

### 2.4 `test_validation_windows_cover_the_stream_without_overlap`

```python
def test_validation_windows_cover_the_stream_without_overlap() -> None:
    tokens = np.arange(100, dtype="<u2")
    assert validation_windows(tokens, 10) == 9
    xs = torch.cat([x for x, _ in validation_batches(tokens, 10, 4)])
    assert xs.shape == (9, 10)
    assert torch.equal(xs.flatten(), torch.arange(90))
```
- 100 tokens, block 10 → 9 windows. Collect all `x` parts from the batches of 4 (4 + 4 + 1 windows) with `torch.cat` → shape `(9, 10)`.
- Flattened, they must equal `0, 1, …, 89` — **no gaps, no overlap, in order**. (Tokens 90–99 are the unused leftover.)

---

## 3. A tiny end-to-end run

### 3.1 The helpers

```python
def _make_repo(root: Path) -> None:
    rng = np.random.default_rng(0)
    tok_dir = root / "data/tokens"
    for split, n in (("train", 300), ("val", 40)):
        lines = [" ".join(map(str, rng.integers(4, VOCAB, size=rng.integers(3, 12)))) for _ in range(n)]
        tokens, offsets = encode_lines(lines, lambda s: [int(t) for t in s.split()], 1, 2)
        meta = TokenFileMeta(FORMAT_VERSION, split, VOCAB, "test", 1, 2, len(lines), len(tokens),
                             sha256_bytes(tokens.tobytes()), "test")
        write_token_files(tok_dir, split, tokens, offsets, meta)
    (root / "configs").mkdir()
    ModelConfig(vocab_size=VOCAB, block_size=16, n_layer=2, n_head=2, d_model=32, name="test").save(
        root / "configs/model_test.json")
```
- Builds a **complete mini project** inside a temporary folder, with no real data needed:
  - 300 training "questions" and 40 validation "questions", each a line of **3 to 11 random numbers** between 4 and 39 (fake token IDs) as text like `"17 5 33"`;
  - `encode_lines(lines, lambda s: [int(t) for t in s.split()], 1, 2)` — the Day 7.1 function, with a tiny **custom "encoder"** (`lambda` = one-line function) that just turns the text into integers; `1` = `<bos>`, `2` = `<eos>`;
  - a meta label (dummy fingerprint `"test"`) and the three files per split via `write_token_files`;
  - a tiny model config (2 layers, width 32, block 16) saved to `configs/model_test.json`.

```python
def _cfg(name: str, **kw) -> TrainConfig:
    base = dict(run_dir=f"runs/{name}", model_config="configs/model_test.json", tokenizer=None, prompts=[],
                batch_size=4, max_steps=12, warmup_steps=3, lr=3e-3, log_every=1, eval_every=4, device="cpu",
                precision="fp32", seed=7)
    return TrainConfig(**{**base, **kw})
```
- Settings for a **12-step** run on the **CPU in fp32**: batch 4, warm-up 3, evaluate every 4 steps, **log every step** (`log_every=1`), `tokenizer=None` (skip the tokenizer check and sample texts), `prompts=[]`. Overrides are merged with `{**base, **kw}` (same trick as before).

```python
def _losses(root: Path, name: str) -> tuple[dict[int, str], dict[int, str]]:
    with (root / "runs" / name / "log.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return ({int(r["step"]): r["train_loss"] for r in rows if r["event"] == "train"},
            {int(r["step"]): r["val_loss"] for r in rows if r["event"] == "eval"})
```
- Read a run's `log.csv` and return two dictionaries **step → loss text** (train, validation). The losses stay as the **strings** that were written, so comparing them means "equal to the last printed digit".

### 3.2 `test_stopped_and_resumed_run_equals_uninterrupted_run` — the key test

```python
def test_stopped_and_resumed_run_equals_uninterrupted_run(tmp_path: Path) -> None:
    _make_repo(tmp_path)
    full = train(_cfg("a"), tmp_path)
    assert full["step"] == 12 and not full["stopped_early"]

    part = train(_cfg("b", stop_after=6), tmp_path)          # 6 is not an evaluation step: last.pt saved at stop
    assert part["step"] == 6 and part["stopped_early"]
    saved = json.loads((tmp_path / "runs/b/train_config.json").read_text(encoding="utf-8"))
    resumed_cfg = TrainConfig.from_dict(saved)
    resumed_cfg.resume, resumed_cfg.stop_after = True, None
    done = train(resumed_cfg, tmp_path)
    assert done["step"] == 12
```
1. Build the mini project in a temp folder (`tmp_path`).
2. **Run "a"**: all 12 steps without stopping; check it finished at step 12.
3. **Run "b"**: stop at step 6 (`stop_after=6`). Step 6 is **not** an evaluation step (evaluations are at 4, 8, 12), so this also exercises "save `last.pt` at the stop". Check it stopped early at step 6.
4. **Resume b**: load b's saved settings, set `resume = True` and clear `stop_after`, run again → it must reach step 12.

```python
    train_a, val_a = _losses(tmp_path, "a")
    train_b, val_b = _losses(tmp_path, "b")
    assert train_a == train_b                                   # every step's loss, to the last printed digit
    assert val_a == val_b
    assert float(val_a[12]) < float(val_a[0])                   # and it learned something
    assert float(val_a[0]) == pytest.approx(math.log(VOCAB), abs=0.3)
    assert (tmp_path / "runs/b/best.pt").exists() and (tmp_path / "runs/b/last.pt").exists()
```
- **Every training loss and every validation loss must be identical** between a and b (dictionary equality on the strings).
- Sanity checks that the test is meaningful: the final validation loss is **lower than the starting one** (it learned), the starting loss is about `ln(40)` (the untrained model is a uniform guesser), and both checkpoint files exist.
- Because it runs on the CPU in fp32 with `log_every=1`, equality is exact.

### 3.3 `test_resume_refuses_changed_settings`

```python
def test_resume_refuses_changed_settings(tmp_path: Path) -> None:
    _make_repo(tmp_path)
    train(_cfg("c", stop_after=4), tmp_path)
    changed = _cfg("c", lr=1e-2, resume=True)
    with pytest.raises(SystemExit):
        train(changed, tmp_path)
```
- Start run "c" and stop at step 4. Then try to resume with a **different learning rate** (`1e-2`). `train` must refuse (`SystemExit`) — see Day 9.2 §5.3 (settings not in `RESUME_FREE_KEYS` must match).

### 3.4 `test_new_run_refuses_an_existing_folder`

```python
def test_new_run_refuses_an_existing_folder(tmp_path: Path) -> None:
    _make_repo(tmp_path)
    train(_cfg("d", max_steps=2, eval_every=2), tmp_path)
    with pytest.raises(SystemExit):
        train(_cfg("d", max_steps=2, eval_every=2), tmp_path)
```
- Run "d" (2 steps). Starting **another new run with the same name** must be refused, so an old run is never overwritten by accident.

### 3.5 `test_resume_after_a_hard_kill_removes_lines_written_after_the_checkpoint`

```python
def test_resume_after_a_hard_kill_removes_lines_written_after_the_checkpoint(tmp_path: Path) -> None:
    _make_repo(tmp_path)
    train(_cfg("a"), tmp_path)
    train(_cfg("e", stop_after=6), tmp_path)
    log = tmp_path / "runs/e/log.csv"
    with log.open("a", encoding="utf-8", newline="") as f:      # lines a killed run would have left behind
        f.write("7,train,448,0.001,9.9,9.9,,,1.0,1,,1.0\n8,eval,512,0.001,,,9.9,9.9,,,,1.0\n")
    cfg = TrainConfig.from_dict(json.loads((tmp_path / "runs/e/train_config.json").read_text(encoding="utf-8")))
    cfg.resume, cfg.stop_after = True, None
    train(cfg, tmp_path)
    assert _losses(tmp_path, "a") == _losses(tmp_path, "e")
```
- Simulates a **hard kill** (window closed, not Ctrl+C):
  - run "a" as the reference; run "e" stopped at step 6 (checkpoint at step 6);
  - **append fake lines** for steps 7 and 8 with junk values (`9.9`) to e's `log.csv` — as if the killed program had logged steps after its last checkpoint (each fake line has the 12 columns of `LOG_COLUMNS`);
  - resume e to the end.
- After resuming, `_truncate_after` (Day 9.2 §4.8) must have **deleted the fake lines**, and the retrained steps must produce the **same logs as the uninterrupted run "a"**. (If the fake `9.9` lines had survived, the dictionaries would differ.)

---

## 4. How many tests?

8 tests: lr schedule (1), weight-decay groups (1), random windows (1), validation windows (1), stop-and-resume equality (1), refuse changed settings (1), refuse existing folder (1), hard kill clean-up (1). Expected: `8 passed`.

## 5. What these tests do *not* cover

- GPU / bf16 / fp16 / the gradient scaler (the tests run on the CPU in fp32).
- Gradient accumulation (`grad_accum > 1`) and gradient clipping directly.
- The Ctrl+C handler (`_StopFlag`) — `stop_after` is used to simulate a stop instead.
- Sample-text generation and the tokenizer check (`tokenizer=None` here).
- The `diverged.pt` safety stop, and `--overwrite`.

## 6. Quick summary in one paragraph

The 8 CPU tests check the learning-rate schedule, that weight decay is applied only to matrices, that training windows are shifted by one and their random state can be saved and restored, that validation windows tile the stream exactly, and — on a tiny fake project built in a temporary folder — that a stopped-and-resumed run produces exactly the same losses as an uninterrupted one, that resuming with changed settings or starting a new run in an existing folder is refused, and that after a hard kill the stale log lines are removed on resume.
