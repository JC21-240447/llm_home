# Day 13.1 — evaluation.py — Simple Line-by-Line Explanation (English)

> **How the four Day 13 files fit together**
> - **Day 13.1 `evaluation.py` (this file)** — the **toolbox**: prompt list, problem detectors, bigram baseline, agreement/kappa, Excel scoring sheet. **Read this first.**
> - **Day 13 `day13_final_eval.py`** — part A: runs the final automatic evaluation and creates the scoring sheet (uses this toolbox).
> - **Day 13 `day13_score_summary.py`** — part B: reads the humans' scores and writes the final evaluation table (uses this toolbox).
> - **Day 13.2 `test_evaluation.py`** — 7 automatic tests of this toolbox.
>
> Suggested reading order: **13.1 → 13 final_eval → 13 score_summary → 13.2**.

---

## 0. The big picture (read this first)

Days 9–12 measured the model with **loss** and a few generation checks. Day 13 is the **final exam**: everything the task 6 document asks for. Some items cannot be measured by generating text alone, so this file adds four tools:

| # | Tool | What it answers |
|---|---|---|
| 1 | Fixed list of ~50 beginnings (6 kinds) | "Which prompts do we test with, always the same?" |
| 2 | Problem detectors (wrong unit, broken number, unseen value) | "Does the model write things like *勤続年数が300万円* (years of service in yen)?" |
| 3 | Bigram counting baseline | "How much better is the Transformer than plain counting?" |
| 4 | Human scoring (Excel sheet, reading it back, agreement, kappa) | "Do real people think the sentences are good, and do they agree with each other?" |

---

## 1. Words you need to know

| Word | Simple meaning |
|---|---|
| **Baseline** | A simple method to compare against. If your model barely beats it, it is not impressive. |
| **Bigram** | A pair of neighbouring tokens. A bigram model predicts the next token from the **previous token only**, just by counting. |
| **Regular expression (regex)** | A pattern for finding text, e.g. "digits followed by 円". |
| **Cohen's kappa** | Agreement between two scorers, corrected for agreement that would happen by luck. 1 = perfect, 0 = no better than chance. |
| **Data validation (Excel)** | A rule in a cell, e.g. "only whole numbers 1–5 allowed". |
| **Calibration** | Checking a detector on data that is known to be good (it should flag ~0). |

---

## 2. Imports

```python
import csv
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
```
- `csv` reads the prompt list and `columns.csv`; `re` = regular expressions; `dataclass`/`field` make small record classes (Day 6.1); `Iterable`/`Sequence` are type hints; `numpy` for the counting table. `openpyxl` (Excel) is imported **inside** the two workbook functions, so the rest works even without it.

---

## 3. Section 1 — the fixed beginnings

```python
PROMPT_KINDS: dict[str, tuple[str, str, int]] = {
    "dept": ("Department or branch", "部署・勤務地で始まる", 10),
    "item": ("Item", "項目で始まる", 10),
    "count_rank": ("Count or rank", "人数・順位を聞く", 10),
    "unseen": ("Wording not in training", "学習にない言い方", 10),
    "key_term": ("Key term variant", "用語とその言い換え", 5),
    "out_of_scope": ("Out of scope", "対象外", 5),
}
```
- A dictionary: **kind → (English name, Japanese name, how many beginnings task 6 Table 7 asks for)**. 10+10+10+10+5+5 = **50**.
- `unseen` = wordings that are not in the training questions (tests generalisation); `out_of_scope` = beginnings the model should not really know (e.g. 今日は).

```python
@dataclass(frozen=True)
class EvalPrompt:
    kind: str
    prompt: str
```
- One test beginning. `frozen=True` = cannot be changed after creation.

```python
def load_eval_prompts(path: Path) -> list[EvalPrompt]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = [EvalPrompt(r["kind"].strip(), r["prompt"].strip()) for r in csv.DictReader(f) if r["prompt"].strip()]
    unknown = {r.kind for r in rows} - set(PROMPT_KINDS)
    if unknown:
        raise ValueError(f"unknown prompt kinds: {sorted(unknown)}")
    if len({r.prompt for r in rows}) != len(rows):
        raise ValueError("a beginning appears twice")
    return rows
```
- Read the CSV (columns `kind`, `prompt`). `utf-8-sig` = UTF-8 that tolerates the invisible mark Excel adds at the start. Blank prompts are skipped; spaces are trimmed.
- **Two safety checks:** every `kind` must be a known one (typos stop the program), and **no beginning may appear twice** (the results are stored by prompt, duplicates would overwrite each other).

---

## 4. Section 2 — the problem detectors

### 4.1 The number-and-unit pattern

```python
_NUMBER_UNIT = r"([0-9][0-9,]*(?:万[0-9]*)?)\s*(円|歳|年|時間|つ|%|％)"
```
A regex with **two groups**:
- Group 1 = the **number**: a digit, then more digits/commas, optionally `万` and more digits (so `800`, `1,000`, `300万`, `1万5000`).
- Group 2 = the **unit**: 円, 歳, 年, 時間, つ, %, ％ (spaces in between are allowed: `\s*`).

### 4.2 `term_phrases`

```python
def term_phrases(columns_csv: Path) -> list[str]:
    phrases: set[str] = set()
    with columns_csv.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            phrases.add(r["列名"].strip())
            for col in ("質問で使う言い方", "平均と一緒に使う言い方"):
                phrases.update(p.strip() for p in r.get(col, "").split("、") if p.strip())
    return sorted(phrases, key=lambda p: (-len(p), p))
```
- Collects **every way the training questions name an item** from `columns.csv`: the column name (列名) plus the alternative phrasings (separated by `、`). A `set` removes duplicates.
- Returned **longest first** (`-len(p)`), then alphabetically. Why longest first? So that `月平均残業時間` is matched before the shorter `残業時間`.

### 4.3 `UnitChecker`

```python
class UnitChecker:
    def __init__(self, phrases: Sequence[str]) -> None:
        alts = "|".join(re.escape(p) for p in sorted(phrases, key=lambda p: (-len(p), p)))
        self.pattern = re.compile(rf"({alts})(?:が|は)?{_NUMBER_UNIT}")
        self.allowed: dict[str, set[str]] = {}
        self.values: set[str] = set()
        self._value = re.compile(_NUMBER_UNIT)
```
- Builds one big pattern: **(item name)(optional が/は)(number)(unit)** — e.g. `年収が800万円`. `re.escape` makes sure special characters in names are taken literally; `|` means "or"; the longest names come first so they win.
- `allowed`: item → set of units seen after it in training (filled by `learn`). `values`: every "number+unit" string seen in training. `_value`: a pattern for number+unit alone.

```python
    @staticmethod
    def _unit(u: str) -> str:
        return "%" if u == "％" else u
```
- Treat the full-width `％` and normal `%` as the same unit.

```python
    def pairs(self, text: str) -> list[tuple[str, str]]:
        return [(m.group(1), self._unit(m.group(3))) for m in self.pattern.finditer(text)]
```
- Find every (item, unit) pair in a text. Group 1 = item, group 2 = number, group 3 = unit.

```python
    def learn(self, lines: Iterable[str]) -> "UnitChecker":
        for line in lines:
            for phrase, unit in self.pairs(line):
                self.allowed.setdefault(phrase, set()).add(unit)
            self.values.update(self.value_strings(line))
        return self
```
- **Learns from the training questions** (no hand-typed rules): for each pair, add the unit to that item's allowed set (`setdefault` creates the empty set the first time); also remember every number+unit string. `return self` lets you write `UnitChecker(...).learn(lines)` in one line.

```python
    def value_strings(self, text: str) -> list[str]:
        return [m.group(1) + self._unit(m.group(2)) for m in self._value.finditer(text)]

    def unseen_values(self, text: str) -> list[str]:
        return [v for v in self.value_strings(text) if v not in self.values]

    def mismatches(self, text: str) -> list[tuple[str, str]]:
        return [(p, u) for p, u in self.pairs(text) if p in self.allowed and u not in self.allowed[p]]
```
- `value_strings` — all "number+unit" strings, e.g. `["800万円", "30歳"]`.
- `unseen_values` — those **no training question contains** (reference check: `30000万円` = 300 million yen was never in the fixed value lists, so it is invented).
- `mismatches` — **wrong unit**: the item is known **and** this unit never followed it in training (`勤続年数 + 円`). Items never seen with any unit are **not judged** (no evidence either way).

### 4.4 Broken numbers

```python
def max_digit_run(lines: Iterable[str]) -> int:
    return max((len(m) for line in lines for m in re.findall(r"[0-9]+", line)), default=0)

def has_broken_number(text: str, limit: int) -> bool:
    return any(len(m) > limit for m in re.findall(r"[0-9]+", text))
```
- `max_digit_run` — the **longest run of digits** in the training questions (e.g. 6 for `社員番号100003`). `re.findall(r"[0-9]+", …)` lists every digit run; `default=0` handles empty input.
- `has_broken_number` — True if the text contains a run **longer than that limit** (e.g. `30000100`). The limit is measured from the data, not guessed.

---

## 5. Section 3 — the bigram baseline

```python
@dataclass
class BigramBaseline:
    pair_counts: np.ndarray      # (V, V)
    row_totals: np.ndarray       # (V,)
    unigram: np.ndarray          # (V,)
    lam: float = 0.9
```
- The model is a **table**: `pair_counts[a][b]` = how often token `b` follows token `a` in the training text. `row_totals[a]` = how often `a` appears as a "previous" token. `unigram` = overall probability of each token. `lam` = how much to trust the pair table (0.9 = 90 %).
- The formula: **P(next | previous) = lam × count(prev,next)/count(prev) + (1 − lam) × P_unigram(next)**. The unigram part uses **add-one counts**, so no token ever gets probability 0 (otherwise the loss would be infinite).

```python
    @classmethod
    def fit(cls, tokens: np.ndarray, vocab_size: int) -> "BigramBaseline":
        t = np.asarray(tokens, dtype=np.int64)
        pairs = np.bincount(t[:-1] * vocab_size + t[1:], minlength=vocab_size * vocab_size)
        pair_counts = pairs.reshape(vocab_size, vocab_size).astype(np.float64)
        uni = np.bincount(t, minlength=vocab_size).astype(np.float64) + 1.0
        return cls(pair_counts, pair_counts.sum(axis=1), uni / uni.sum())
```
- Convert to `int64` (the token file is `uint16`; multiplying could overflow).
- **A trick to count all pairs at once:** each pair (a, b) becomes one number `a × V + b`; `np.bincount` counts how often each number appears; `reshape(V, V)` turns the counts into the table.
- `uni` = single-token counts **+ 1** (add-one), divided by the total → probabilities. `pair_counts.sum(axis=1)` = the row totals.
- Size: 1,674 × 1,674 floats ≈ 22 MB — fine.

```python
    def loss(self, tokens: np.ndarray, lam: float | None = None) -> float:
        lam = self.lam if lam is None else lam
        t = np.asarray(tokens, dtype=np.int64)
        prev, nxt = t[:-1], t[1:]
        totals = self.row_totals[prev]
        p_bi = np.divide(self.pair_counts[prev, nxt], totals, out=np.zeros(len(prev)), where=totals > 0)
        w = np.where(totals > 0, lam, 0.0)
        p = w * p_bi + (1.0 - w) * self.unigram[nxt]
        return float(-np.log(p).mean())
```
- For every (previous, next) pair in the stream: look up its count and the previous token's total; divide **only where the total > 0** (`where=`; others stay 0).
- `w` = `lam` where the previous token was seen in training, else **0** (then only the unigram is used).
- Mix the two parts, take **−log**, average → the **same kind of loss** as the Transformer's (natural log), so they can be compared directly (and `exp(loss)` = perplexity).

```python
    def tune(self, val_tokens, grid=(0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99)) -> float:
        self.lam = min(grid, key=lambda g: self.loss(val_tokens, g))
        return self.lam
```
- Try each `lam`, keep the one with the **lowest validation loss**. **Never the test stream** — the test text stays untouched until the final numbers.

---

## 6. Section 4 — agreement between scorers

```python
def agreement_within_one(a, b) -> float:
    if len(a) != len(b) or not a:
        raise ValueError("need two equally long, non-empty score lists")
    return sum(abs(x - y) <= 1 for x, y in zip(a, b)) / len(a)
```
- Share of outputs where the two scorers differ by **at most 1 point** (the task 6 guideline: ≥ 70 %). `True` counts as 1 in `sum`.

```python
def cohen_kappa(a, b, labels=(1, 2, 3, 4, 5)) -> float:
    ...
    n = len(a)
    p_o = sum(x == y for x, y in zip(a, b)) / n
    p_e = sum((a.count(k) / n) * (b.count(k) / n) for k in labels)
    if p_e >= 1.0:
        return 1.0 if p_o >= 1.0 else 0.0
    return (p_o - p_e) / (1.0 - p_e)
```
- `p_o` = observed **exact** agreement. `p_e` = agreement expected by **chance**: for each score, (share of A giving it) × (share of B giving it), added up.
- **kappa = (p_o − p_e) / (1 − p_e).** Example: if both always say "5", p_e = 1 → division by zero, so the special case returns 1.0 if they really agree, else 0.0.
- Note: `a.count(k)` needs lists (not numpy arrays); the callers pass lists.

---

## 7. Section 5 — the scoring workbook

### 7.1 Constants and records

```python
H_ITEMS = ("H1", "H2", "H3", "H4")
PROBLEM_TYPES = {"G": ("Grammar error", "文法の誤り"), "R": ("Repetition", "繰り返し"), "T": ("Wrong term use", "用語の誤り"),
                 "O": ("Off topic", "話題がずれる"), "U": ("Unfinished", "途中で終わる"), "N": ("Invented name", "存在しない名前")}
```
- H1 correct Japanese, H2 natural, H3 makes sense, H4 company terms (task 6 Table 8); six problem-type codes (Table 9).
- `_HEADERS` = column titles (bilingual); `_CRITERIA` = the 1–5 scale for each item, shown on the instructions sheet so every scorer uses the same meaning of "3".

```python
@dataclass
class ScoringRow:  id: str; prompt: str; text: str; out_of_scope: bool
@dataclass
class Score:       h: dict[str, int | None]; types: set[str] = field(default_factory=set); note: str = ""
class ScoreFileError(ValueError): pass
```
- `ScoringRow` = one line to be scored (what the scorer sees). `Score` = what one scorer gave for one line: the four scores (`None` = not scored), problem codes, a note. `field(default_factory=set)` gives each Score its **own** empty set. `ScoreFileError` = a special error type for bad score files.

### 7.2 `write_scoring_workbook`

- Creates an Excel file with **two sheets**:
  1. **"説明 Instructions"** — rules (score alone, whole numbers 1–5, skip H4 for out-of-scope rows, problem codes, "…" = unfinished, rows are shuffled), the criteria table, the code list. Only the table wraps text.
  2. **"採点 Scoring"** — one row per output: `ID, beginning, continuation, H1, H2, H3, H4, problem types, note`. Header row blue and bold. For out-of-scope rows, **H4 is pre-filled with "-" and greyed**.
- **Data validation:** columns D–F (H1–H3) accept **only whole numbers 1–5** (an error popup appears otherwise); H4 (column G) has the rule only on rows that are in scope.
- Column widths, frozen panes (`D2` keeps ID/beginning/continuation visible while scrolling), wrapped text, and `wb.active = 1` opens the scoring sheet first.

### 7.3 `read_scores` — read one scorer's file, **collect all problems**

- Opens the scorer's file (`data_only=True` = values, not formulas), goes through each row and checks:
  - the ID is one of the handed-out IDs;
  - **the beginning and text were not changed** compared with the original sheet;
  - each H score is present and a **whole number 1–5** (`3.5` is refused: `isinstance(value, float) and value != v`); H4 is skipped for out-of-scope rows (stored as `None`);
  - problem codes are valid: split on commas/spaces (`[,、\s]+`), upper-cased (`t` → `T`), `-` ignored, unknown codes reported;
  - no rows are missing.
- **All problems are collected first and raised together** (first 40 shown), so a scorer can fix everything in one go instead of one error per run.

### 7.4 `mean`

```python
def mean(values): v = list(values); return sum(v) / len(v) if v else math.nan
```
- Average; returns `nan` ("not a number") for an empty list instead of crashing.

---

## 8. Small things worth knowing

- **Wrong unit only judges known items**: an item that never had a unit in training is skipped, so the detector is cautious (few false alarms) but not complete.
- **Detectors look at prompt + continuation** (the caller passes the whole sentence), so a value written in the *prompt* that is not in training can show up as "unseen value".
- **Unseen value is only a reference**: a new value is not always wrong.
- **The bigram table has 1,674² numbers**; it is built from the **training** tokens only and tuned on **validation**.
- **The `％`/`%` fix** avoids counting the same unit as two different ones.
- **kappa can look low even when agreement is high** if almost all scores are 4–5 (the "kappa paradox") — read the two numbers together (Day 13 score_summary says so in its report).
- **Excel may change text** (for example trailing spaces); `read_scores` compares text exactly, so scorers must not edit columns B–C.

## 9. Quick summary in one paragraph

`evaluation.py` is the Day 13 toolbox: the fixed list of 50 beginnings in 6 kinds (with safety checks), detectors for wrong units, broken numbers and unseen values whose rules are **learned from the training questions**, a counting (bigram) baseline for perplexity tuned on validation only, the within-1-point agreement and Cohen's kappa for pairs of scorers, and an Excel scoring sheet (1–5 enforced) with a reader that checks every score file and reports all its problems at once.
