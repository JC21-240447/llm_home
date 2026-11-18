# Day 12.3 — test_training.py (updated) — What Changed + Line-by-Line Explanation (English)

> **This is the UPDATED version of `test_training.py`** that you first saw on Day 9.3.
> - **The first 8 tests are identical.** Their full explanation is in **`day9.3_test_training_explained_EN.md`** (and the JA file).
> - **Two new tests + one new helper were added for early stopping.** This file explains them.
>
> **How the five Day 12 files fit together:** `day12_train_main.py`, `day12_monitor.py`, `generation.py` (12.1), `training.py` (12.2 — the code with the new early stopping), **`test_training.py` (12.3, this file — tests it)**.

---

## 0. The big picture (read this first)

Early stopping must satisfy two promises:

1. **It really stops the run, and keeps the best model** when validation stops improving.
2. **It still works after a stop-and-resume** — the run must stop at the **same step** as a run that was never interrupted.

To test (1) we need a situation where validation loss is **guaranteed to get worse**. Real data cannot promise that, so the test builds a **rigged mini-dataset** on purpose.

### The rigged data (why validation must get worse)

| | Token IDs used |
|---|---|
| **Training text** | only **4 – 9** (6 different tokens) |
| **Validation text** | only **20 – 39** (never seen in training) |

The model quickly learns "only tokens 4–9 exist" and gives tokens 20–39 almost **zero probability**. So **training loss falls, but validation loss goes UP**. The best validation point is at the very start, and after 6 more steps without improvement early stopping **must** trigger.

---

## 1. Unchanged parts

The imports, `VOCAB = 40`, the helpers `_make_repo`, `_cfg`, `_losses` and the first 8 tests are exactly as in Day 9.3 (learning-rate schedule, weight-decay groups, random windows, validation windows, stop-and-resume equality, refusing changed settings, refusing an existing folder, cleanup after a hard kill).

---

## 2. The new helper: `_make_diverging_repo`

```python
def _make_diverging_repo(root: Path) -> None:
    """Training text uses only token IDs 4-9, validation only 20-39: learning the training text makes validation
    worse, so the validation loss is sure to rise and early stopping must trigger."""
    rng = np.random.default_rng(1)
    tok_dir = root / "data/tokens"
    for split, lo, hi, n in (("train", 4, 10, 300), ("val", 20, 40, 40)):
        lines = [" ".join(map(str, rng.integers(lo, hi, size=8))) for _ in range(n)]
        tokens, offsets = encode_lines(lines, lambda s: [int(t) for t in s.split()], 1, 2)
        meta = TokenFileMeta(FORMAT_VERSION, split, VOCAB, "test", 1, 2, len(lines), len(tokens),
                             sha256_bytes(tokens.tobytes()), "test")
        write_token_files(tok_dir, split, tokens, offsets, meta)
    (root / "configs").mkdir()
    ModelConfig(vocab_size=VOCAB, block_size=16, n_layer=2, n_head=2, d_model=32, name="test").save(
        root / "configs/model_test.json")
```
- Same idea as `_make_repo` (Day 9.3), with a different data recipe:
  - the loop gives, for each split, the **range of token IDs** `lo` to `hi` (the upper end is exclusive: `rng.integers(4, 10)` gives 4…9) and the number of lines `n`: **train** = IDs 4–9, 300 lines; **val** = IDs 20–39, 40 lines;
  - each line = **8 random IDs** written as text (`"7 5 9 4 …"`); a tiny custom "encoder" turns the text into integers; `1` = `<bos>`, `2` = `<eos>`;
  - the usual meta label and files, plus a small model config (2 layers, width 32, block 16).
- `np.random.default_rng(1)` — a fixed seed, so the fake data is the same every time.

---

## 3. New test 1: early stopping ends the run and keeps the best

```python
def test_early_stopping_ends_the_run_and_keeps_the_best(tmp_path: Path) -> None:
    """Day 12: with a patience, training stops once validation has not improved for that many steps."""
    _make_diverging_repo(tmp_path)
    res = train(_cfg("f", max_steps=60, eval_every=2, lr=1e-2, early_stop_patience=6), tmp_path)
    status = json.loads((tmp_path / "runs/f/status.json").read_text(encoding="utf-8"))
    assert res["early_stopped"] and status["status"] == "early_stopped"
    assert res["step"] < 60 and res["step"] - res["best_step"] >= 6
    _, val = _losses(tmp_path, "f")
    assert float(val[res["best_step"]]) == pytest.approx(res["best_val"], abs=1e-4)
    assert min(float(v) for v in val.values()) == pytest.approx(res["best_val"], abs=1e-4)
```
- Build the rigged data. Train with: a 60-step schedule, evaluate **every 2 steps**, a large learning rate `1e-2` (so the model learns the training text fast), **patience 6**.
- `status = …` reads the `status.json` written by Day 12.2.
- **Check 1:** the result says `early_stopped`, and `status.json` agrees (`"early_stopped"`).
- **Check 2:** it stopped **before** step 60, and at least **6 steps after the best step** (`res["step"] - res["best_step"] >= 6`) — exactly the rule.
- **Check 3 (the best is really the best):** read the logged validation losses (`_losses` returns train and validation dictionaries; `_` ignores the train one). The loss logged at `best_step` must equal the **recorded best value** `res["best_val"]`, and the **smallest** validation loss in the whole log must also equal it. `pytest.approx(..., abs=1e-4)` allows a tiny difference because the log stores losses as text with 4 decimals.

---

## 4. New test 2: early stopping survives stop-and-resume

```python
def test_early_stopping_survives_stop_and_resume(tmp_path: Path) -> None:
    """The best step is saved in the checkpoint, so a resumed run stops at the same step as a non-stop run."""
    _make_diverging_repo(tmp_path)
    full = train(_cfg("g", max_steps=60, eval_every=2, lr=1e-2, early_stop_patience=6), tmp_path)
    train(_cfg("h", max_steps=60, eval_every=2, lr=1e-2, early_stop_patience=6, stop_after=full["step"] - 3),
          tmp_path)
    cfg = TrainConfig.from_dict(json.loads((tmp_path / "runs/h/train_config.json").read_text(encoding="utf-8")))
    cfg.resume, cfg.stop_after = True, None
    resumed = train(cfg, tmp_path)
    assert (resumed["step"], resumed["best_step"]) == (full["step"], full["best_step"])
    assert _losses(tmp_path, "g") == _losses(tmp_path, "h")
```
- **Run "g"** — the **uninterrupted** reference: trains until early stopping triggers at some step `full["step"]`.
- **Run "h"** — the **same settings**, but **deliberately interrupted** 3 steps before that point (`stop_after = full["step"] - 3`). Because `full["step"]` is an evaluation step (even), minus 3 is an **odd** step — *not* an evaluation step — so this also exercises "save `last.pt` at the stop" (Day 9.2).
- **Resume "h"**: load its saved settings, set `resume = True`, clear `stop_after`, train again.
- **Check 1:** the resumed run ends at the **same step** and with the **same `best_step`** as the uninterrupted run.
- **Check 2:** **all logged training and validation losses are identical** between g and h.
- **Why this catches a bug:** if the checkpoint did **not** store `best_step`, the resumed run would restart the counter (`best_step` = the step where it resumed, via the fallback in Day 12.2 §2.4) and early stopping would trigger **later** than in run g — the check would fail. So this test proves `best_step` is saved and restored correctly.

---

## 5. How many tests?

**10 tests** now (8 from Day 9.3 + 2 new). Expected: `10 passed`. All on the CPU in a few seconds.

## 6. What these tests do *not* cover

- Early stopping when **`early_stop_patience` is `None`** (the default) — covered indirectly by the 8 old tests, which never set it.
- Loading an **old checkpoint** without `best_step` (the `.get` fallback).
- The **early-stop boundary** exactly (`>=`): the test only checks that the stop happens at least 6 steps after the best.
- `status.json` for `finished` and `stopped` (only `early_stopped` is asserted).
- `day12_monitor.py` and `day12_train_main.py` themselves (they need real data/models).

## 7. Quick summary in one paragraph

Two new CPU tests, built on a rigged mini-dataset where validation is guaranteed to get worse (training uses token IDs 4–9, validation 20–39), prove that early stopping stops the run, writes `status.json = early_stopped`, keeps the true best validation loss, stops at least `patience` steps after the best point, and — after a deliberate stop and resume — ends at exactly the same step, with the same best step and identical logs, as an uninterrupted run.
