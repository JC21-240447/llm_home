# Day 13.2 — test_evaluation.py — Simple Line-by-Line Explanation (English)

> **How the four Day 13 files fit together**
> - **Day 13.1 `evaluation.py`** — the code being tested (**read first**).
> - **Day 13 `day13_final_eval.py`** (part A) and **`day13_score_summary.py`** (part B) — the two drivers.
> - **Day 13.2 `test_evaluation.py` (this file)** — **7 automatic tests** of the toolbox. CPU only, a few seconds.
>
> pytest basics were explained on Days 6.2–12.3. Only new things are explained here.

---

## 0. The big picture (read this first)

The final evaluation produces the numbers you will **defend in front of the faculty**. These tests prove, on tiny hand-checkable examples, that the tools behave as promised:

| Promise | Test |
|---|---|
| The prompt list matches task 6 Table 7 (50 beginnings, counts per kind) | `test_eval_prompt_list_matches_task6_table7` |
| Units are **learned** from training questions; the longest name wins | `test_units_are_learned_from_the_training_questions` |
| Wrong units are flagged; correct ones (and the training questions themselves) are not | `test_wrong_unit_is_flagged_and_correct_units_are_not` |
| The broken-number limit comes from the training data | `test_broken_number_uses_the_longest_number_in_training` |
| The bigram baseline is a proper probability distribution and beats uniform | `test_bigram_baseline_is_a_proper_distribution_and_beats_uniform` |
| Agreement and kappa give the known, hand-calculated values | `test_agreement_and_kappa_known_values` |
| The Excel sheet round-trips, and bad score files are reported **all at once** | `test_scoring_workbook_round_trip_and_problem_report` |

Run: `.venv\Scripts\python.exe -m pytest tests -q`

---

## 1. Setup

```python
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from src.evaluation import (PROMPT_KINDS, BigramBaseline, ScoreFileError, ScoringRow, UnitChecker,
                            agreement_within_one, cohen_kappa, has_broken_number, load_eval_prompts, max_digit_run,
                            read_scores, write_scoring_workbook)

REPO_ROOT = Path(__file__).resolve().parent.parent
PHRASES = ["月平均残業時間", "残業時間", "勤続年数", "年収", "年齢", "賞与を除いた年収"]
TRAIN = ["年収が800万円以上の社員は？", "勤続年数が10年以上の人", "年齢が30歳未満の社員",
         "月平均残業時間が20時間を超える人", "賞与を除いた年収が500万円以上の社員", "社員番号100003の人"]
```
- Import the tools from **Day 13.1**. (No `sys.path` line: this test file assumes pytest finds `src` from the project root, as configured earlier.)
- `PHRASES` — item names as in `columns.csv`; note that some names **contain others** (`残業時間` inside `月平均残業時間`, `年収` inside `賞与を除いた年収`) on purpose. `TRAIN` — six tiny "training questions" covering four units and a long number (社員番号100003).

---

## 2. The tests

### 2.1 `test_eval_prompt_list_matches_task6_table7`

```python
    prompts = load_eval_prompts(REPO_ROOT / "prompts/day13_eval_prompts.csv")
    assert len(prompts) == 50
    for kind, (_, _, count) in PROMPT_KINDS.items():
        assert sum(p.kind == kind for p in prompts) == count
```
- The only test that reads a **real project file**: the real `prompts/day13_eval_prompts.csv` must have exactly **50** beginnings, and **each kind must have the count task 6 asks for** (10/10/10/10/5/5). Loading also runs the safety checks (known kinds, no duplicates).

### 2.2 `test_units_are_learned_from_the_training_questions`

```python
    uc = UnitChecker(PHRASES).learn(TRAIN)
    assert uc.allowed["年収"] == {"円"} and uc.allowed["勤続年数"] == {"年"} and uc.allowed["年齢"] == {"歳"}
    assert "月平均残業時間" in uc.allowed and "残業時間" not in uc.allowed
    assert "賞与を除いた年収" in uc.allowed
```
- After learning from `TRAIN`: 年収 → only 円, 勤続年数 → only 年, 年齢 → only 歳 (nobody typed these rules).
- **Longest name wins:** `月平均残業時間が20時間` is stored under `月平均残業時間`, **not** under the shorter `残業時間`; likewise `賞与を除いた年収` is not read as plain `年収`. (If the short name had won, a detector would compare units with the wrong item.)

### 2.3 `test_wrong_unit_is_flagged_and_correct_units_are_not`

```python
    assert uc.mismatches("勤続年数が300万円以上の社員を教えてください。") == [("勤続年数", "円")]
    assert uc.mismatches("年齢が30年以上の社員") == [("年齢", "年")]
    assert uc.mismatches("年収が700万円未満で勤続年数が5年以上の人") == []
    assert uc.mismatches("残業時間が10時間") == []          # never seen with a unit in training: not judged
    assert all(uc.mismatches(q) == [] for q in TRAIN)       # no false alarms on the training questions
    assert uc.unseen_values("年収が30000万円以下の社員") == ["30000万円"]
    assert uc.unseen_values("年収が800万円以上で年齢が30歳未満") == []
```
- Years of service in yen → **flagged**. Age in years → **flagged**. A sentence with two correct pairs → **nothing flagged**.
- `残業時間が10時間` → not flagged, because `残業時間` was **never learned** (it only appears inside the longer name): "items never seen are not judged".
- **No false alarms** on any training question — the same idea as the calibration on real test questions in part A.
- `30000万円` was never in training → listed as an **unseen value**; `800万円` and `30歳` were → empty list.

### 2.4 `test_broken_number_uses_the_longest_number_in_training`

```python
    limit = max_digit_run(TRAIN)
    assert limit == 6                                        # 社員番号100003
    assert has_broken_number("ボーナスが30000100万円未満の社員は？", limit)
    assert not has_broken_number("年収が800万円以上", limit)
    assert not has_broken_number("社員番号100003の人", limit)
```
- The longest digit run in `TRAIN` is `100003` → **6**. `30000100` has 8 digits → **broken**. `800` is fine. `100003` (exactly 6, a real employee number) is **not** broken (the rule is "longer than the limit").

### 2.5 `test_bigram_baseline_is_a_proper_distribution_and_beats_uniform`

```python
    rng = np.random.default_rng(0)
    vocab = 12
    stream = np.array([1, 4, 5, 6, 2] * 200 + list(rng.integers(3, vocab, 200)), dtype=np.uint16)
    b = BigramBaseline.fit(stream, vocab)
    lam = b.tune(stream[:300])
    for prev in range(vocab):
        p = [math.exp(-b.loss(np.array([prev, nxt]), lam)) for nxt in range(vocab)]
        assert sum(p) == pytest.approx(1.0)
    assert b.loss(stream[:500]) < math.log(vocab)
```
- A fake token stream: a pattern `1 4 5 6 2` repeated 200 times (very predictable) plus 200 random tokens; vocabulary of 12.
- **Proper distribution:** for **every** previous token, the probabilities of all 12 possible next tokens must **add up to 1**. The trick: the loss of a 2-token stream `[prev, nxt]` is exactly `−log P(nxt | prev)`, so `exp(−loss)` is the probability. (This also proves the add-one unigram and the `lam` mixing are correct, including for previous tokens never seen.)
- **Beats uniform:** on the predictable stream the loss is below `ln(12)` — the uniform guess.

### 2.6 `test_agreement_and_kappa_known_values`

```python
    a = [5, 4, 3, 2, 1, 5]
    assert agreement_within_one(a, a) == 1.0 and cohen_kappa(a, a) == pytest.approx(1.0)
    b = [4, 4, 1, 2, 3, 5]               # differences 1, 0, 2, 0, 2, 0
    assert agreement_within_one(a, b) == pytest.approx(4 / 6)
    p_e = 7 / 36
    assert cohen_kappa(a, b) == pytest.approx((0.5 - p_e) / (1 - p_e))      # = 11/29
```
- Identical scorers → agreement 1.0, kappa 1.0.
- Scorer `b` differs from `a` by 1, 0, 2, 0, 2, 0 → within 1 point in **4 of 6** cases.
- **Kappa by hand:** exact agreement = 3/6 = 0.5 (positions 2, 4, 6). Chance agreement (share of a × share of b, summed over scores): score 5 → 2/6 × 1/6, score 4 → 1/6 × 2/6, scores 3, 2, 1 → 1/6 × 1/6 each → **7/36**. kappa = (0.5 − 7/36) / (1 − 7/36) = (11/36) / (29/36) = **11/29 ≈ 0.38**. The test writes the hand calculation in the comments, so the code is checked against **independent arithmetic**.

### 2.7 `test_scoring_workbook_round_trip_and_problem_report`

```python
    openpyxl = pytest.importorskip("openpyxl")
    rows = [ScoringRow("S001", "年収が", "高い社員は？", False), ScoringRow("S002", "今日は", "何人いますか。", True)]
    sheet = tmp_path / "scoring_sheet.xlsx"
    write_scoring_workbook(sheet, rows)
```
- `pytest.importorskip("openpyxl")` — if openpyxl is not installed, this test is **skipped** (not failed). Two rows, the second out of scope. `tmp_path` = pytest's temporary folder.

```python
    wb = openpyxl.load_workbook(sheet)
    ws = wb["採点 Scoring"]
    assert ws["G3"].value == "-"                         # out of scope: H4 is not scored
    for col, v in zip("DEFG", (5, 4, 5, 4)):
        ws[f"{col}2"] = v
    for col, v in zip("DEF", (3, 2, 1)):
        ws[f"{col}3"] = v
    ws["H3"] = "O, t"
    filled = tmp_path / "scores_ab.xlsx"
    wb.save(filled)
    got = read_scores(filled, rows)
    assert got["S001"].h == {"H1": 5, "H2": 4, "H3": 5, "H4": 4}
    assert got["S002"].h["H4"] is None and got["S002"].types == {"O", "T"}
```
- Open the blank sheet; the out-of-scope row already has `-` in H4. **Act as a scorer:** fill H1–H4 for row 2, H1–H3 for row 3, and problem codes `"O, t"` (messy on purpose: a space and a lowercase letter) for row 3. Save as `scores_ab.xlsx` and **read it back**.
- Row S001 comes back as exactly 5, 4, 5, 4; S002's H4 is `None` and its codes are `{"O", "T"}` (the lowercase `t` was fixed to `T`).

```python
    ws["E2"] = None                                      # a missing score, a bad value and a bad code
    ws["F3"] = 7
    ws["H2"] = "X"
    wb.save(filled)
    with pytest.raises(ScoreFileError) as e:
        read_scores(filled, rows)
    msg = str(e.value)
    assert "S001: H2 is empty" in msg and "S002: H3 must be a whole number 1-5" in msg and "['X']" in msg
```
- **Break the file on purpose with three different mistakes:** S001's H2 deleted, S002's H3 set to 7, S001's problem code set to `X`. Reading must raise `ScoreFileError` — and the **one message must mention all three** (that is the "all problems at once" promise).

---

## 3. How many tests?

**7 tests**. Expected: `7 passed` (or `6 passed, 1 skipped` if openpyxl is missing), in about a second.

## 4. What these tests do *not* check

- The **exact** boundary of the broken-number or unit rules on real data (the calibration on the real test questions in part A does that).
- `term_phrases` reading the real `columns.csv`.
- Cohen's kappa for the special case `p_e = 1` (everyone always gives the same score).
- The **look** of the Excel sheet (colours, widths, data-validation popups): only values are checked.
- Score files where the **text was edited** (`the beginning or text differs`), or the 40-problem display limit.
- `day13_final_eval.py` and `day13_score_summary.py` themselves (they need the real model and scores).

## 5. Quick summary in one paragraph

These 7 quick tests prove that the real prompt list matches task 6 (50 beginnings, right counts per kind); that units are learned from training questions with the longest name winning; that wrong units and invented values are flagged without false alarms; that the broken-number limit is taken from the data; that the bigram baseline is a true probability distribution better than uniform guessing; that agreement and kappa match hand calculations (kappa = 11/29); and that the Excel scoring sheet round-trips and a damaged score file produces one message listing every problem.
