# Day 13 — day13_final_eval.py — Simple Line-by-Line Explanation (English)

> **How the four Day 13 files fit together**
> - **Day 13.1 `evaluation.py`** — the toolbox (read first).
> - **Day 13 (this file) `day13_final_eval.py`** — **part A**: the final automatic evaluation of the final model + the human scoring sheet.
> - **Day 13 `day13_score_summary.py`** — part B: reads the human scores, writes the final table.
> - **Day 13.2 `test_evaluation.py`** — tests of the toolbox.
>
> Suggested reading order: **13.1 → 13 final_eval (this) → 13 score_summary → 13.2**.

---

## 0. The big picture (read this first)

Day 12 chose the final model (`configs/day12_final.json`) using **validation** text only. Day 13 is the **final exam**:

| Part | What | Why |
|---|---|---|
| **A1 perplexity** | on validation **and the TEST text** (first and only use of the test text) | an honest number: no choice was ever made with it |
| **References** | uniform guessing, untrained model, bigram counting baseline | a number alone means nothing; "compared with what?" |
| **A2–A5** | finished / repetitive / distinct-2 / copy on **50 fixed beginnings**, 2 generation methods, **best.pt and last.pt** | task 6 checklist |
| **Extra checks** | wrong unit, broken number, unseen value | the errors Day 12 showed (e.g. 勤続年数が300万円) |
| **Human scoring sheet** | the final model's 100 outputs, shuffled, method hidden | humans judge Japanese quality (part B) |

```
.venv\Scripts\python.exe scripts\day13_final_eval.py --machine home
```

### What is written

| File | Content |
|---|---|
| `docs/DAY13_AUTO_EVALUATION.md` | the automatic results (report) |
| `docs/figures/day13_perplexity.png` | A1 compared with the references |
| `docs/day13/auto_results.json` | all numbers (part B reads it) |
| `docs/day13/generations.csv` | all **200** outputs with their checks (opens in Excel) |
| `docs/day13/scoring_sheet.xlsx` | the sheet for the scorers |
| `docs/day13/scoring_key.json` | which method made each scored output (**hide from scorers**) |

---

## 1. Words you need to know

| Word | Simple meaning |
|---|---|
| **Test text** | Data never used for any decision. Using it once at the end gives an honest score. |
| **Perplexity** | `exp(loss)`: "how many tokens the model is torn between". Lower = better. |
| **Blind scoring** | Scorers do not know which method made an output, so they cannot be biased. |
| **Shuffle** | Random re-ordering (here with a fixed seed 13, so it is reproducible). |
| **A2–A5** | Task 6 automatic items: A2 finished, A3 repetitive, A4 distinct-2, A5 copy of a training question. |

---

## 2. Imports and constants

```python
import argparse, csv, json, math, platform, random, sys
from datetime import datetime
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
from src.batching import open_tokens
from src.bpe_tokenizer import ByteBPETokenizer
from src.evaluation import (PROMPT_KINDS, BigramBaseline, ScoringRow, UnitChecker, has_broken_number,
                            load_eval_prompts, max_digit_run, term_phrases, write_scoring_workbook)   # Day 13.1
from src.generation import (EOS_ID, PRESET_SETTINGS, distinct_2, generate, is_repetitive, load_model, model_logits_fn)
from src.model import ModelA, ModelConfig
from src.selection import MAX_REPETITIVE, MIN_FINISHED
from src.training import evaluate, pick_precision

SCRIPT_VERSION = "day13-v1"
METHODS = ("greedy+limits", "sample")
SHUFFLE_SEED = 13
```
- New standard module: **`random`** (for the shuffle). From the project: token files (7.1/9.1), tokenizer (6.1), the **Day 13.1 toolbox**, generation (10.1), model (8.1), the guideline thresholds (11.1), and `evaluate` + `pick_precision` from the training library (9.2) — the **same** loss function as in training, so numbers are comparable.
- `METHODS` — the two generation methods compared (presets from Day 10.1). `SHUFFLE_SEED = 13` — fixes the shuffle.

---

## 3. `parse_args` and `read_lines`

- Options: `--machine`; `--final` (default `configs/day12_final.json`, the hand-over file from Day 12); `--prompts` (the 50 beginnings CSV); `--device`; `--batch-size 32`; `--eval-max-windows` (testing only); `--out-dir docs/day13`; `--report`; `--figure`; **`--force`** (replace the scoring sheet even if scores already exist); `--no-log`.
- `read_lines(path)` — the file's lines **without empty ones**.

---

## 4. `main`

### 4.1 Find the final model

```python
    final = json.loads(a.final.read_text(encoding="utf-8"))
    best_path = REPO_ROOT / final["final_checkpoint"]
    run_dir = best_path.parent
    last_path = run_dir / "last.pt"
    for p in (best_path, last_path):
        if not p.exists():
            raise SystemExit(f"{p} not found. Day 13 needs the Day 12 run folder ({run_dir}); do not delete runs/day12.")
```
- Read Day 12's hand-over file → path of `best.pt`; `last.pt` is in the same folder. If either is missing, stop with a clear message (the `runs/day12` folder is git-ignored, so it is easy to delete by accident).

```python
    device = ...                       # same nested conditional as before
    precision = pick_precision("auto", device)
    mcfg = ModelConfig.load(REPO_ROOT / final["model_config"])
    vocab = mcfg.vocab_size
```
- Device, precision (bf16/fp16/fp32 as in training), model settings; `vocab` = 1,674.

### 4.2 Data

```python
    tok = ByteBPETokenizer.load(...)
    train_tok, val_tok, test_tok = (open_tokens(tokens_dir, s) for s in ("train", "val", "test"))
    train_lines = read_lines(REPO_ROOT / "data/corpus/train.txt")
    test_lines = read_lines(REPO_ROOT / "data/corpus/test.txt")
    train_set = set(train_lines)
    prompts = load_eval_prompts(a.prompts)
    counts = {k: sum(p.kind == k for p in prompts) for k in PROMPT_KINDS}
```
- Token streams of the three splits (**this is where `test` is opened for the first time**), the raw training and test questions (as text), `train_set` (fast lookup for "copy"), the 50 beginnings, and the count per kind (printed).

### 4.3 `loss_of`

```python
    def loss_of(model, tokens) -> float:
        v = evaluate(model, tokens, mcfg.block_size, a.batch_size, device, precision, a.eval_max_windows)
        model.eval()
        return v
```
- Average loss of a model on a token stream, using Day 9.2's `evaluate`. That function switches the model **back to training mode** at its end, so `model.eval()` turns off dropout again — **generation must run in eval mode**.

### 4.4 A1 — perplexity and the references

```python
    a1["uniform"]   = {"label": ..., "val": math.log(vocab), "test": math.log(vocab)}
```
- **Uniform guessing:** every token equally likely → loss = ln(1,674) ≈ 7.42 (perplexity = 1,674).

```python
    torch.manual_seed(0)
    untrained = ModelA(mcfg).to(device).eval()
    a1["untrained"] = {... "val": loss_of(untrained, val_tok), "test": loss_of(untrained, test_tok)}
    del untrained
```
- **Untrained model:** same size, random weights (fixed seed 0) → shows where training started. `del` frees memory.

```python
    bigram = BigramBaseline.fit(train_tok, vocab)
    lam = bigram.tune(val_tok)
    a1["bigram"] = {..., "val": bigram.loss(val_tok), "test": bigram.loss(test_tok)}
```
- **Counting baseline (Day 13.1):** fit on **training** tokens, weight `lam` chosen on **validation**, then reported on both.

```python
    models = {}
    for name, path in (("best.pt", best_path), ("last.pt", last_path)):
        model, info = load_model(path, device)
        models[name] = (model, info)
        a1[name] = {"label": f"Final model {name} (step {info['step']})", "val": ..., "test": ..., "step": info["step"]}
```
- Load **both** checkpoints of the final run and measure each. `best.pt` = lowest validation loss; `last.pt` = where training ended (task 6, 6-2 asks for both).

```python
    others = {}
    for seed in final.get("seeds", []):
        p = run_dir.parent / f"main_s{seed}" / "best.pt"
        if p.exists() and p.resolve() != best_path.resolve():
            m, info = load_model(p, device)
            others[f"main_s{seed}"] = {...}
            del m
```
- The **other two seeds**' `best.pt` also get validation and test numbers — **only as a reference** (how much the test result depends on the seed). They are evaluated **after** the final model was fixed, so they played no part in any choice.
- Then everything is printed: `loss (perplexity)` for validation and test.

### 4.5 Detectors, checked on real data first

```python
    unit_checker = UnitChecker(term_phrases(REPO_ROOT / "data/source/columns.csv")).learn(train_lines)
    digit_limit = max_digit_run(train_lines)
    calib = {"questions": len(test_lines),
             "wrong_unit": sum(bool(unit_checker.mismatches(q)) for q in test_lines),
             "broken_number": sum(has_broken_number(q, digit_limit) for q in test_lines),
             "unseen_value": sum(bool(unit_checker.unseen_values(q)) for q in test_lines),
             "digit_limit": digit_limit, "items_learned": len(unit_checker.allowed)}
```
- Build the unit checker (learns from training questions) and measure the digit limit. **Calibration:** run each detector on all the **real test questions** — real questions should produce ~0 flags. If they do, a flag on a generated sentence is a **real problem, not a false alarm** (this reasoning is written in the report).

```python
    unseen = [p.prompt for p in prompts if p.kind == "unseen"]
    unseen_new = sum(not any(q.startswith(u) for q in train_lines) for u in unseen)
```
- Of the "wording not in training" beginnings, how many **really never start** a training question (checks that the 10 "unseen" prompts are truly unseen).

### 4.6 A2–A5 on the 50 beginnings

```python
    for mname, (model, _) in models.items():          # best.pt, last.pt
        fn = model_logits_fn(model, device)
        for method in METHODS:                          # greedy+limits, sample
            body_ids, rows = [], []
            for p in prompts:                           # 50 beginnings
                g = generate(fn, tok, p.prompt, PRESET_SETTINGS[method])
                body = [i for i in g.ids if i != EOS_ID]
                body_ids.append(body)
                full = p.prompt + g.text
                rows.append({"model": mname, "method": method, "kind": p.kind, "prompt": p.prompt, "text": g.text,
                    "finished": g.finished, "repetitive": is_repetitive(tok.encode(p.prompt) + body),
                    "copy": g.finished and full in train_set,
                    "wrong_unit": "; ".join(f"{t}+{u}" for t, u in unit_checker.mismatches(full)),
                    "broken_number": has_broken_number(full, digit_limit),
                    "unseen_value": "; ".join(unit_checker.unseen_values(full))})
```
- **2 checkpoints × 2 methods × 50 beginnings = 200 outputs.** For each: generate (max 40 tokens, seed 42, as in the presets); `body` = generated IDs without `<eos>`; `full` = beginning + continuation (the detectors look at the whole sentence).
- Per output: **finished** (ended with `<eos>`), **repetitive** (more repetition than any training question), **copy** (finished **and** exactly a training question — A5, a reference: memorising), **wrong_unit** / **unseen_value** (text listing the findings, empty = none), **broken_number**.

```python
            n = len(rows)
            agg[f"{mname}|{method}"] = {"finished": sum(...)/n, "repetitive": ..., "distinct2": distinct_2(body_ids),
                "copy": ..., "wrong_unit": sum(bool(r["wrong_unit"]) ...)/n, "broken_number": ..., "unseen_value": ...}
            outputs.extend(rows)
```
- Summarise each (checkpoint, method) as **fractions** (distinct-2 is computed over all outputs together), keep all rows, print one line per combination.

### 4.7 The human scoring sheet

```python
    scored = [r for r in outputs if r["model"] == "best.pt"]          # 100 outputs
    random.Random(SHUFFLE_SEED).shuffle(scored)
    key = [{"id": f"S{i:03d}", "method": r["method"], "kind": r["kind"], "prompt": r["prompt"],
            "text": r["text"] + ("" if r["finished"] else " …"), "out_of_scope": r["kind"] == "out_of_scope"}
           for i, r in enumerate(scored, 1)]
```
- Only the **final model (best.pt)**, both methods → **100 outputs** (so each scorer reads 100, not 200). `random.Random(13).shuffle` mixes greedy and sample outputs in a **reproducible random order**, so scorers cannot guess the method.
- `key` = the **answer key**: ID `S001…S100` (zero-padded), method, kind, beginning, text (an unfinished output gets " …"), and whether it is out of scope.

```python
    key_path, sheet_path = a.out_dir / "scoring_key.json", a.out_dir / "scoring_sheet.xlsx"
    scores_dir = a.out_dir / "scores"
    has_scores = scores_dir.exists() and any(scores_dir.glob("scores_*.xlsx"))
    old_key = json.loads(key_path.read_text(encoding="utf-8"))["rows"] if key_path.exists() else None
    if has_scores and old_key is not None and old_key != key and not a.force:
        raise SystemExit("The new outputs differ from the sheet the scorers already used ... Use --force only if you intend to score again.")
```
- **A safety guard:** if scorers have **already put score files** in `docs/day13/scores`, and the new outputs **differ** from the old key (for example because the model or GPU gave slightly different text), **stop without overwriting** — otherwise the scores would no longer match the key. `--force` overrides on purpose.

```python
    key_path.write_text(json.dumps({...,"shuffle_seed": SHUFFLE_SEED, "rows": key}, ...))
    write_scoring_workbook(sheet_path, [ScoringRow(k["id"], k["prompt"], k["text"], k["out_of_scope"]) for k in key])
    scores_dir.mkdir(exist_ok=True)
```
- Write the key JSON, create the Excel sheet (**method column is not in it**), make the empty `scores/` folder where scorers put their files.

### 4.8 Files, figure, report, log

- `generations.csv` — all 200 rows. **`utf-8-sig`** = UTF-8 with the mark Excel needs to show Japanese correctly.
- `auto_results.json` — script, date, machine, device, the final-model info, A1, other seeds, aggregated generation numbers, calibration, prompt counts, unseen count, the method settings, number of scored outputs, the guidelines. **Part B reads this file.**
- `draw_figure`, `write_report`, then an entry in `run_log.md` (test loss/perplexity, bigram baseline, one line per checkpoint-method, sheet info). `return 0`.

---

## 5. `draw_figure`

- A bar chart of **test perplexity** for: uniform, untrained, bigram, last.pt, best.pt, on a **log scale** (otherwise the 1,674 bar would make the others invisible). Grey bars = references, blue = the final model. The value is printed above each bar (`v * 1.12` puts it slightly higher). `set_ylim(1, max × 3)` leaves room for the labels. **Needs matplotlib.**

## 6. `_cell` and `write_report`

`_cell` makes text safe for a Markdown table (as Day 12). `write_report` builds `docs/DAY13_AUTO_EVALUATION.md`:

| Section | Content |
|---|---|
| Header | version, date, machine; the final model; "**the test text is used here for the first time**" |
| **A1 Perplexity** | figure + table (validation/test loss and perplexity for every row, other seeds marked "(reference)"); comparison with the untrained model and with the bigram model ("shows what the Transformer adds by reading the whole beginning") |
| **A2–A5** | generation settings and guidelines (finished ≥ 90 %, repetitive ≤ 5 %); one table row per checkpoint × method |
| Calibration note | what the detectors found on the real test questions; the digit limit; the explanation of "unseen value" |
| By kind | per kind: count, finished / wrong unit / copied for greedy vs sample (final model) |
| **All outputs** | every beginning with the greedy and sampling continuation of the final model; flags **U** wrong unit, **B** broken number, **V** unseen value, **R** repetitive, **C** exact copy, **…** did not finish |
| Next | where the scoring sheet is, how scorers should save their copy (`docs/day13/scores/scores_<initials>.xlsx`) and what part B does; **last.pt is checked automatically only** |

---

## 7. The whole run in order

1. Read Day 12's hand-over file; check `best.pt` and `last.pt` exist.
2. Load data; A1 perplexity for references, best.pt, last.pt, other seeds.
3. Calibrate the detectors on the real test questions.
4. Generate 200 outputs; compute A2–A5 and extra checks.
5. Shuffle the final model's 100 outputs → key + Excel sheet (with the overwrite guard).
6. CSV, JSON, figure, report, run log.

## 8. How to read the results

- **A1:** the test perplexity of `best.pt` should be **far below** untrained (≈ 1,674) and **below the bigram** number. Test close to validation = no overfitting to the validation choices.
- **best vs last:** if `last.pt` is much worse, the model overfitted late in training (this is why Day 12 kept `best.pt`).
- **Finished / repetitive:** compare with the guidelines (≥ 90 % / ≤ 5 %).
- **Wrong unit / broken number:** the problems Day 12 revealed; ideally near 0 %.
- **Copy:** a high copy rate means memorising training questions.

## 9. Small things worth knowing

- **The test text is evaluated for every row of the A1 table**, including the untrained model and the other seeds. That is fine because none of these results changed any decision — but do **not** use the test numbers to pick anything now.
- **Re-running** gives the same text on the same machine (greedy and seeded sampling are deterministic), so the scoring key stays identical; on a different GPU tiny differences are possible, which is exactly what the guard is for.
- **Wrong-unit detection only judges items that appeared with a unit in training.**
- **"seed 42"** and "max 40 tokens" are written in the report text, not read from the settings.
- **Unseen value** also looks at numbers written in the beginning itself.
- **200 generations on CPU can be slow** (one forward pass per new token).
- **Needs matplotlib and openpyxl.**
- **Do not show `scoring_key.json` to scorers** — it reveals which output came from which method.

## 10. Quick summary in one paragraph

`day13_final_eval.py` is part A of the final exam: it measures the Day 12 final model (best.pt and last.pt) on the validation **and, for the first time, the test text**, next to uniform, untrained and bigram references; generates 200 outputs from 50 fixed beginnings with two methods and computes finished, repetitive, distinct-2, copy and three extra detectors (calibrated on real test questions); then shuffles the final model's 100 outputs into a blinded Excel scoring sheet with a separate answer key (and a guard against overwriting a sheet scorers already used), and writes the report, figure, CSV, JSON and run log.
