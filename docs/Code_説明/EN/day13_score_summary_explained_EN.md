# Day 13 — day13_score_summary.py — Simple Line-by-Line Explanation (English)

> **How the four Day 13 files fit together**
> - **Day 13.1 `evaluation.py`** — the toolbox (read first).
> - **Day 13 `day13_final_eval.py`** — part A: automatic evaluation + scoring sheet.
> - **Day 13 (this file) `day13_score_summary.py`** — **part B**: reads the humans' scores and writes the **final evaluation table** (completes task 11).
> - **Day 13.2 `test_evaluation.py`** — tests of the toolbox.
>
> Suggested reading order: **13.1 → 13 final_eval → 13 score_summary (this) → 13.2**.

---

## 0. The big picture (read this first)

After part A, two or more people each filled in a copy of the Excel sheet. Part B:

1. **Reads** every `docs/day13/scores/scores_<initials>.xlsx` and **checks** them (all scores present, whole numbers 1–5, valid codes, text unchanged) — reporting **all** problems at once.
2. **Calculates** (task 6, sections 6-4 and 7-3): H1–H4 averages per method; share of outputs with H3 ≤ 2; agreement between scorers (within 1 point, Cohen's kappa); problem types.
3. **Writes** the **final evaluation table** against the pass guidelines (task 6 Table 10), including part A's numbers: `docs/DAY13_EVALUATION.md` and `docs/figures/day13_human_scores.png`.

```
.venv\Scripts\python.exe scripts\day13_score_summary.py
```

---

## 1. Words you need to know

| Word | Simple meaning |
|---|---|
| **H1–H4** | The four human scores (1–5): H1 correct Japanese, H2 natural, H3 makes sense, H4 company terms. |
| **Scorer** | A person who fills in the sheet. |
| **Within 1 point** | Two scorers differ by at most 1 (e.g. 4 and 5). Guideline: ≥ 70 %. |
| **Cohen's kappa** | Exact agreement minus luck-agreement (Day 13.1). 1 = perfect, 0 = chance. |
| **Guideline** | The pass levels written in task 6 **before** the evaluation. They are not changed afterwards. |

---

## 2. Imports and constants

```python
import argparse, itertools, json, math, sys
from datetime import datetime
from pathlib import Path
...
from src.evaluation import (H_ITEMS, PROBLEM_TYPES, PROMPT_KINDS, Score, ScoreFileError, ScoringRow,
                            agreement_within_one, cohen_kappa, mean, read_scores)      # Day 13.1

SCRIPT_VERSION = "day13-v1"
MIN_H_MEAN = 3.5
MAX_H3_LOW = 0.10
MIN_AGREEMENT = 0.70
H_NAMES = {"H1": "H1 correct Japanese", "H2": "H2 natural", "H3": "H3 makes sense", "H4": "H4 company terms"}
```
- New standard module: **`itertools`** (`combinations` = all pairs of scorers).
- The **pass guidelines** are constants: every H mean ≥ **3.5**; outputs with H3 ≤ 2 at most **10 %**; agreement within 1 point at least **70 %** per item. (Finished ≥ 90 % and repetitive ≤ 5 % come from part A's saved guidelines.)
- `H_NAMES` — display names for the report and the figure.

---

## 3. `parse_args` and `yes_no`

- Options: `--scores-dir` (default `docs/day13/scores`), `--key` (the answer key), `--auto` (part A's JSON), `--report`, `--figure`, `--no-log`.
- `yes_no(ok)` → `"-"` if `None` (not judged), `**yes**` or `**no**` (bold in Markdown).

---

## 4. `main`

### 4.1 Read and check all score files

```python
    key = json.loads(a.key.read_text(encoding="utf-8"))["rows"]
    auto = json.loads(a.auto.read_text(encoding="utf-8"))
    expected = [ScoringRow(k["id"], k["prompt"], k["text"], k["out_of_scope"]) for k in key]
    files = sorted(a.scores_dir.glob("scores_*.xlsx"))
    if not files:
        raise SystemExit("No scores_*.xlsx in ... Each scorer copies docs/day13/scoring_sheet.xlsx there as scores_<initials>.xlsx and fills it in.")
```
- Load the answer key and part A's results; turn the key into the list of rows that were **handed out** (used to check the files). Find the score files (sorted); none → stop with instructions.

```python
    scorers: dict[str, dict[str, Score]] = {}
    errors: list[str] = []
    for f in files:
        try:
            scorers[f.stem.removeprefix("scores_")] = read_scores(f, expected)
        except ScoreFileError as e:
            errors.append(str(e))
    if errors:
        print("The score files have problems; nothing was written. Fix these and run again:\n")
        print("\n\n".join(errors))
        return 1
```
- For each file, `read_scores` (Day 13.1) returns `{output id → Score}`. The scorer's name = the file name without `scores_` (`scores_ab.xlsx` → `ab`).
- If **any** file has problems, print **all** of them (from **every** file) and **stop with nothing written**, return code 1. So the final table is never based on incomplete scores.
- `names` = list of scorers. **With only one scorer a WARNING is shown** (agreement cannot be measured).

### 4.2 Each output's score = mean over scorers

```python
    per_out = {}
    for k in key:
        per_out[k["id"]] = {}
        for h in H_ITEMS:
            vals = [scorers[n][k["id"]].h[h] for n in names if scorers[n][k["id"]].h[h] is not None]
            per_out[k["id"]][h] = mean(vals) if vals else None
```
- For each output and each H item: the **average of all scorers' scores** (e.g. 4 and 5 → 4.5). `None` scores (H4 of out-of-scope outputs) are skipped; if nobody scored it → `None`.

```python
    def h_means(rows): return {h: mean(per_out[r["id"]][h] for r in rows if per_out[r["id"]][h] is not None) for h in H_ITEMS}
    def h3_low(rows):  return sum(per_out[r["id"]]["H3"] <= 2 for r in rows) / len(rows)
    groups = {"greedy+limits": [...], "sample": [...], "both": key}
    means = {g: h_means(rows) for g, rows in groups.items()}
    lows  = {g: h3_low(rows) for g, rows in groups.items()}
```
- `h_means` — average of the per-output means for one group of rows; `h3_low` — the share of outputs whose (mean) H3 is **≤ 2** (nonsense is a failure by itself).
- **Three groups:** greedy outputs (50), sample outputs (50), both (100) — that is why the method had to be hidden from the scorers but is known here (from the key).

### 4.3 Agreement between scorers

```python
    agreement = {}
    if len(names) >= 2:
        for h in H_ITEMS:
            ag, kp = [], []
            for x, y in itertools.combinations(names, 2):
                ids = [k["id"] for k in key if scorers[x][k["id"]].h[h] is not None]
                ax = [scorers[x][i].h[h] for i in ids]
                ay = [scorers[y][i].h[h] for i in ids]
                ag.append(agreement_within_one(ax, ay))
                kp.append(cohen_kappa(ax, ay))
            agreement[h] = {"within_one": mean(ag), "kappa": mean(kp)}
```
- For each H item and **each pair of scorers** (`combinations(names, 2)`: with 3 scorers → 3 pairs): take the outputs where the item was scored, line up both scorers' scores, compute within-1 agreement and kappa. Then **average over the pairs** → one number per item. (With one scorer, `agreement` stays empty.)

### 4.4 Problem types

```python
    types_of = {k["id"]: set().union(*(scorers[n][k["id"]].types for n in names)) for k in key}
    type_counts = {code: sum(code in types_of[k["id"]] for k in key) for code in PROBLEM_TYPES}
```
- For each output: the **union** of the problem codes given by all scorers ("counts if **any** scorer gave the type"). Then, per code (G, R, T, O, U, N), how many outputs have it.

### 4.5 Print, figure, report, log

- Print one line per group: `H1 … H4` means and H3 ≤ 2 share; one line per item for agreement. Then `draw_figure`, `write_report`, and an entry in `run_log.md` (scorers, H means, H3 ≤ 2, agreement or "not measured").

---

## 5. `draw_figure`

- A grouped bar chart: for each of H1–H4, a blue bar (greedy+limits) and an orange bar (sample), the **value above each bar**, and a **red dashed line at the guideline 3.5**. y-axis from 1 to 5.4. Bars below the red line fail.

---

## 6. `write_report` — the final evaluation table

### 6.1 The rows of the table

```python
    g_best = {m: gen[f"best.pt|{m}"] for m in ("greedy+limits", "sample")}
    test_ppl, untrained_ppl, bigram_ppl = exp(test loss of best.pt), exp(untrained), exp(bigram)
    finished_ok = all(v["finished"] >= min_finished ...)
    rep_ok = all(v["repetitive"] <= max_repetitive ...)
    agree_ok = all(v["within_one"] >= MIN_AGREEMENT ...) if agreement else None
```
- Take the **final model's** automatic numbers from part A (`best.pt` for both methods) and check each guideline (**both** methods must pass). `agree_ok` is `None` with one scorer.

| Item | Guideline | How it is judged |
|---|---|---|
| A1 Perplexity (test) | far below the untrained value | **test perplexity < untrained ÷ 10** (the "far below" threshold is this script's own choice) |
| A2 Completion rate | ≥ 90 % | both methods |
| A3 Repetition rate | ≤ 5 % | both methods |
| H1–H4 (mean) | ≥ 3.5 | the mean over **both** methods |
| Outputs with H3 ≤ 2 | ≤ 10 % | over both methods |
| Agreement (within 1 point) | ≥ 70 % per item | every item |

- `passed` = rows that are exactly `True`; `judged` = rows that are not `None`. The report says **"N of M judged items meet the guidelines."**

### 6.2 The sections

| Section | Content |
|---|---|
| Header | version, date, sources (part A report date, number of scorers), final model, "criteria: the task 6 document, **unchanged**" |
| **Final evaluation table** | the rows above; values for generation shown as **greedy / sampling** |
| Human scores | figure, explanation (H4 is not scored for out-of-scope beginnings), table per method (greedy, sample, both) |
| **By kind of beginning** | for each of the 6 kinds: number of outputs, H1–H4 means (`-` where `nan`, e.g. H4 for out-of-scope), share with H3 ≤ 2 |
| **Agreement between scorers** | within-1 and kappa per item + an explanation that **kappa is stricter**, and that when most scores are 4–5 two scorers are often within 1 point by chance. If any kappa < **0.2**, a bold warning: scorers should compare a few outputs where they differed and make the criteria wording clearer |
| **Problem types** | per code (G/R/T/O/U/N): number of outputs, plus the **automatic check from part A** for related types (T ↔ wrong unit %, R ↔ repetitive %, U ↔ unfinished %) |
| **Lowest and highest scored outputs** | the **8 lowest** and **5 highest** outputs by mean over H1–H4 and scorers (ties by ID): method, beginning, text, mean score, problem types and the **scorers' notes** |
| Notes | generation values are for best.pt; the guidelines are not changed here, and a later revision must be recorded with its date |

```python
    def mean_all(i): v = [x for x in per_out[i].values() if x is not None]; return sum(v) / len(v)
    ranked = sorted(key, key=lambda k: (mean_all(k["id"]), k["id"]))
    ... ranked[:8] ... ranked[-5:][::-1]
```
- `mean_all` — one number per output (average of its H1–H4 means). `ranked` = lowest first; `ranked[:8]` = the 8 worst, `ranked[-5:][::-1]` = the 5 best (best first). This lets you read the **actual worst sentences** and the scorers' notes — more useful than averages alone.

---

## 7. The whole run in order

1. Read key + part A results; find the score files.
2. Check every file; if any problem → print them all and stop.
3. Mean over scorers per output; H means for greedy / sample / both; H3 ≤ 2 shares.
4. Agreement (pairs of scorers) and problem-type counts.
5. Figure → final evaluation table report → run log.

## 8. How to read the results

- **The table first:** "N of M items meet the guidelines" and the yes/no column.
- **Which items fail?** Compare H1–H4 with 3.5 (red line in the figure); look at the by-kind table to see **which kind of beginning** is weak (often *unseen* wording and *out of scope*).
- **Agreement:** high within-1 but low kappa = scorers mostly give similar high scores but rarely the same number; discuss the differing outputs.
- **Lowest-scored outputs + notes:** what to fix next (units, repetition, invented terms).

## 9. Small things worth knowing

- **"Mean over scorers" can give 2.5 for H3**, which is **not** ≤ 2, so it does not count as a failure; the rule uses the mean of the scorers, not each person's score.
- **H4 is averaged only over in-scope outputs** (out-of-scope = `None`).
- **The A1 threshold (÷ 10) is not from task 6** — it is this script's reading of "far below". Say so if asked.
- **Problem types are a union** (any scorer), so a single scorer can add a type to an output.
- **The method is revealed only here**, after scoring, through the key.
- **With one scorer**: agreement and the agreement row are "not measured", and the table has fewer judged items.
- **Part B never changes the guidelines** — they were fixed before the evaluation.
- **Needs matplotlib**; the score files need **openpyxl** (through Day 13.1).

## 10. Quick summary in one paragraph

`day13_score_summary.py` reads every scorer's Excel file, checks all of them and stops with a full list of problems if any are wrong; otherwise it averages each output's scores over the scorers, reports H1–H4 for greedy, sampling and both, the share of outputs with H3 ≤ 2, the within-1-point agreement and Cohen's kappa between scorers, the problem types and the lowest/highest scored outputs, and combines them with part A's automatic numbers into the final evaluation table (yes/no against each pass guideline), a figure and a run log entry.
