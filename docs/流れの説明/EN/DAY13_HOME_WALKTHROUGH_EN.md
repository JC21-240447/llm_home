# Day 13 (home laptop): the final evaluation

**Machine:** home laptop (RTX 4060 Laptop). **Repository:** `JC21-240447/llm_home`. **Folder:** `C:\Users\Merrick\Desktop\llm_home`.
**School plan:** task 11 「LLMの学習、動作確認、生成結果の評価」 **completes today**:
- Day 11 = choose the size;
- Day 12 = main training;
- **Day 13 = the evaluation table.**

Task 11 is complete when the final evaluation table (`docs\DAY13_EVALUATION.md`) is written and pushed.

## What Day 13 does, and why

Day 12 produced the final model, `runs\day12\main_s44\best.pt`. Today we evaluate it **exactly as the task 6 document says**. That document was written before any training, so the criteria were not chosen to suit the results.

Day 13 has two parts:

**Part A: measured by a program (about 1 minute).**
1. **A1 perplexity on the test text.** This is the **first and only time** the test text is used. Every earlier choice (size, dropout, seed) used only the validation text, so the test result is an honest measure of unseen text. Three references show what the number means:
   - uniform guessing (about 1,674);
   - the untrained model;
   - a **counting baseline**, the bigram model. It predicts the next token by counting which token followed the previous one in the training text, and it uses nothing else. The gap between it and our model shows what the Transformer gains by reading the whole beginning of the sentence.
2. **A2–A5 on 50 fixed beginnings** (task 6, Table 7):

   | Kind | Count |
   | --- | --- |
   | Department or branch | 10 |
   | Item | 10 |
   | Count or rank | 10 |
   | Wording not in training | 10 |
   | Key term variants | 5 |
   | Out of scope | 5 |

   Generation uses:
   - two methods: greedy + limits, and sampling with a fixed seed;
   - at most 40 tokens;
   - **both** `best.pt` and `last.pt` (task 6, 6-2).

   The 10 "wording not in training" beginnings come from test sentence patterns. The script checks that none of them starts a training question.
3. **Three extra automatic checks** for the errors you saw on Day 12:
   - **Wrong unit:** an item followed by a unit that never follows it in the training questions, e.g. 勤続年数が300万円. The allowed units are **learned from the training questions**, not typed by hand.
   - **Broken number:** a digit run longer than any in the training questions, e.g. 30000100.
   - **Unseen value (reference):** a number with its unit that no training question contains, e.g. 30000万円.

   Before these checks are used on the model, they are run on the 4,553 **real test questions**: all three must find (almost) nothing there. So a flag on a generated sentence is a real problem, not a false alarm. On my copy of the data: 0, 0 and 0.
4. **The scoring sheet** (`docs\day13\scoring_sheet.xlsx`). It holds the 100 outputs of the final model (50 beginnings × 2 methods), **shuffled, with the method hidden** (task 6, 6-4).

**Part B: scored by people.** Two or more team members score the 100 outputs **separately**, in their own copy of the sheet, from 1 to 5:
- H1 correct Japanese;
- H2 natural;
- H3 makes sense;
- H4 company terms.

They also add problem-type codes. Then a second script reads all the copies, checks them, and writes the **final evaluation table** against the task 6 pass guidelines (Table 10), including the agreement between scorers.

### New files (nothing is replaced today)

| File | What |
| --- | --- |
| `src\evaluation.py` | the 50 beginnings loader; the wrong-unit, broken-number and unseen-value checks; the bigram baseline; agreement and Cohen's kappa; writing and reading the Excel scoring sheet |
| `scripts\day13_final_eval.py` | part A |
| `scripts\day13_score_summary.py` | part B: reads the scores, writes the final evaluation table |
| `prompts\day13_eval_prompts.csv` | the 50 fixed beginnings with their kind |
| `tests\test_evaluation.py` | 7 new tests |

### Outputs

| Output | What |
| --- | --- |
| `docs\DAY13_AUTO_EVALUATION.md` | part A: A1 table, A2–A5 and the extra checks, results by kind, every output of the final model |
| `docs\figures\day13_perplexity.png` | test perplexity of the model against the references |
| `docs\day13\scoring_sheet.xlsx` | the sheet to hand to the scorers |
| `docs\day13\scoring_key.json` | which method made each scored output (**do not show the scorers**) |
| `docs\day13\generations.csv` | all 200 outputs with their checks (opens in Excel) |
| `docs\day13\auto_results.json` | part A's numbers; part B reads them |
| `docs\day13\scores\scores_<initials>.xlsx` | one filled-in copy per scorer |
| `docs\DAY13_EVALUATION.md` | **the final evaluation table** (part B) |
| `docs\figures\day13_human_scores.png` | H1–H4 by method |

### SERVER NOTE (later)

Same code: `.venv/bin/python scripts/day13_final_eval.py --machine server`, then `.venv/bin/python scripts/day13_score_summary.py`.

---

## Step 1. Put the files in place

1. Download `day13.zip` and unzip it.
2. Copy **everything inside** the `day13` folder into `llm_home`. All files are new; Windows should not ask to replace anything.
3. Check that the Day 12 run folder is still there (VS Code terminal, `llm_home`):
   ```powershell
   dir runs\day12\main_s44
   ```
   You should see `best.pt` and `last.pt`. Day 13 needs both.

## Step 2. Install the Excel library (VS Code terminal, `llm_home`)

The scoring sheet is an Excel file, written and read with the `openpyxl` library. Install it into the project's virtual environment only:

```powershell
.venv\Scripts\python.exe -m pip install openpyxl
```

If your repository has a `requirements.txt`, add a line `openpyxl` to it, so the server can install the same packages later.

## Step 3. Run all tests (VS Code terminal, `llm_home`)

```powershell
.venv\Scripts\python.exe -m pytest tests -q
```

Expected: **`81 passed`** (74 + 7 new).

| No. | New test | What it proves |
| --- | --- | --- |
| 75 | `test_eval_prompt_list_matches_task6_table7` | the beginnings file has exactly 50 beginnings, with the counts per kind of task 6 Table 7 (10/10/10/10/5/5) |
| 76 | `test_units_are_learned_from_the_training_questions` | the allowed units are learned from questions (年収 → 円, 勤続年数 → 年, 年齢 → 歳), and the longest item name wins (月平均残業時間 is not read as 残業時間) |
| 77 | `test_wrong_unit_is_flagged_and_correct_units_are_not` | 勤続年数が300万円 and 年齢が30年 are flagged; correct questions are not; an item never seen with a unit is not judged; 30000万円 is an unseen value |
| 78 | `test_broken_number_uses_the_longest_number_in_training` | the digit limit is measured from the questions (6, from 社員番号100003); 30000100 is flagged, 100003 is not |
| 79 | `test_bigram_baseline_is_a_proper_distribution_and_beats_uniform` | for every previous token, the counting baseline's probabilities sum to 1, and it does better than uniform guessing |
| 80 | `test_agreement_and_kappa_known_values` | the within-1-point agreement and Cohen's kappa give the values worked out by hand (for example, kappa = 11/29) |
| 81 | `test_scoring_workbook_round_trip_and_problem_report` | the Excel sheet is written and read back correctly; out-of-scope rows skip H4; a missing score, a score of 7 and an unknown code are all reported together, with the row ID |

## Step 4. Run part A (VS Code terminal, `llm_home`)

```powershell
.venv\Scripts\python.exe scripts\day13_final_eval.py --machine home
```

It takes about a minute. What you will see:
1. `Beginnings: 50 (dept 10, item 10, count_rank 10, unseen 10, key_term 5, out_of_scope 5)`.
2. `--- A1 perplexity ---`: five lines (uniform, untrained, bigram, best.pt, last.pt), then the other two seeds as a reference.
3. `Detectors on the 4553 real test questions (should be ~0): ...`.
4. `--- A2-A5 on the fixed beginnings ---`: four lines (best.pt and last.pt, each with two methods).
5. `Scoring sheet: ... (100 outputs)`.

**Do not open `docs\DAY13_AUTO_EVALUATION.md` or `generations.csv` yet.** They show which method made each output, along with the automatic flags. If you read them before scoring, your own scores will be influenced. Score first (Step 6), then read.

## Step 5. Prepare the scoring copies

Task 6 asks for **two or more scorers** who read Japanese: you plus at least one teammate.

1. In File Explorer, open `llm_home\docs\day13`.
2. For each scorer, copy `scoring_sheet.xlsx` and name the copy **`scores_<initials>.xlsx`**, for example `scores_km.xlsx` for you and `scores_ts.xlsx` for a teammate. Keep the copies in your own folder until they come back.
3. Send each teammate **only their own `scores_<initials>.xlsx`**, by Teams, mail or USB. Never send them `scoring_key.json`, `generations.csv` or the part A report.

Rules for every scorer (also on the first sheet of the file, in Japanese and English):
- **score alone**; don't discuss with the others until everyone has finished;
- in the sheet 「採点 Scoring」, enter a whole number **1–5** in H1–H4 (Excel refuses anything else);
- where H4 shows **「-」** (out-of-scope beginnings), leave H4 as it is;
- if there is a problem, enter codes in 「問題の種類」: G grammar, R repetition, T wrong term use, O off topic, U unfinished, N invented name. Several are allowed, e.g. `T,O`;
- optionally, a short note in 「メモ」;
- text ending in 「…」 did not finish;
- do not change the beginnings or the texts. Sorting or filtering the rows is fine, because rows are matched by ID;
- save as **.xlsx** (Ctrl+S). Do not save as CSV.

It takes about **30–45 minutes** per person for 100 outputs.

## Step 6. Score

Score your own copy. The criteria table on the first sheet is task 6, Table 8. Score what is written, not what the model probably meant. Example: 「勤続年数が300万円以上の社員は？」 is grammatical (H1 high), but the item and the unit don't fit (H4 low, code T).

## Step 7. Collect the copies and run part B (VS Code terminal, `llm_home`)

1. Put every filled-in copy into `llm_home\docs\day13\scores\`. The folder already exists.
2. Run:
   ```powershell
   .venv\Scripts\python.exe scripts\day13_score_summary.py
   ```
3. **If a file has a problem**, nothing is written, and every problem is listed at once with the row ID, for example:
   ```
   scores_ts.xlsx: 2 problem(s)
     S004: H1 is empty
     S006: unknown problem type code(s) ['Z'] (use G, R, T, O, U, N)
   ```
   Fix those cells (or ask the scorer to), save, and run the command again.
4. When it succeeds, it prints the H1–H4 averages per method and the agreement per item, and writes `docs\DAY13_EVALUATION.md`.

## Step 8. Read the reports

Now open both reports (press **Ctrl+Shift+V** in VS Code).

**`docs\DAY13_AUTO_EVALUATION.md` (part A):**
1. **A1 table and figure.** Test perplexity of `best.pt` against the three references.
   - It should be far below uniform and untrained (about 1,670).
   - The interesting row is the **bigram** one. On my copy of the data, a stand-in model trained with your Day 12 settings scored test perplexity **11.7** against bigram **12.9**. On validation it was 13.1 against 19.4.
   - So the Transformer is clearly better than counting, but on the test patterns the gain is small. Many test sentences can be predicted from word pairs alone. Whatever your numbers are, report this comparison as it is.
2. **A2–A5 table.** Finished, repetitive, distinct-2, copy, and the three extra checks, for `best.pt` and `last.pt`.
   - On my stand-in, `best.pt` had **wrong unit in about 18%** of outputs. After item beginnings, greedy often wrote 「30000万円…」 whatever the item was.
   - `last.pt` had fewer wrong units but copied training questions more.
   - This is the same lesson as Day 9/10: the lowest validation loss and the best use of company terms are not the same point.
3. **By kind.** Item and key-term beginnings carry most of the unit errors. Department beginnings and wording not in training are cleaner.
4. **All outputs** of the final model, with flags U/B/V/R/C.

**`docs\DAY13_EVALUATION.md` (part B, the final evaluation table):**
1. **Final evaluation table:** each task 6 guideline, the result, and whether it is met, then **"N of M judged items meet the guidelines"**.
2. **Human scores** by method and by kind.
3. **Agreement:** the within-1-point rate (the guideline) **and** Cohen's kappa. If kappa is below 0.2, the report says so.
   - When most scores are 4 or 5, two people are often within 1 point by chance alone, so kappa is the stricter check.
   - Low kappa means the scorers should compare a few outputs where they differed. That would be a finding about the criteria, not a failure of the model.
4. **Problem types**, next to the automatic checks. Compare the human "T wrong term use" count with the automatic wrong-unit rate: they should point the same way.
5. **Lowest and highest scored outputs**, with the scorers' notes.

Not every guideline has to be met. If H4 or the wrong-unit rate is weak, that is the result, and it is exactly what the next task (improvement) starts from. The guidelines are **not changed** after seeing the results.

## Step 9. Record and commit

Add to `docs\DECISIONS.md` with your numbers:

```markdown
## D13. Final evaluation (Day 13, completes task 11)

- Final model runs/day12/main_s44/best.pt, evaluated with the task 6 criteria (unchanged).
- A1 test perplexity <x.xx> (validation <x.xx>); untrained <n>; bigram counting baseline <x.xx>.
  The test text was used here for the first time.
- 50 fixed beginnings x 2 methods: finished <x> % / <x> %, repetitive <x> % / <x> %, distinct-2 <x.xx> / <x.xx>.
  Extra checks (0 false alarms on the real test questions): wrong unit <x> % / <x> %, broken number <x> % / <x> %,
  unseen value <x> % / <x> %. last.pt: wrong unit <x> % / <x> %, copy <x> % / <x> %.
- Human scoring (<n> scorers, 100 outputs, method hidden): H1 <x.xx>, H2 <x.xx>, H3 <x.xx>, H4 <x.xx>;
  H3 <= 2: <x> %; agreement within 1 point <x>-<x> %, kappa <x.xx>-<x.xx>.
- <N> of <M> guidelines met. Main weakness: <e.g. item + unit combinations (wrong unit x %, H4 x.xx)>.
- Task 11 complete.
```

Then commit (VS Code terminal, `llm_home`):

```powershell
git status --short
git add src scripts tests prompts configs docs run_log.md
git status --short
git commit -m "Day 13 home: final evaluation (task 6 criteria), human scoring, task 11 complete"
git push
```

`docs\day13\` goes into Git, including the scores. The scorers' file names contain only initials. Nothing under `runs/` and no `.pt` file may appear in the list.

## Talking points for the review

- **Why the test text was used only now:** every choice was made with the validation text. If the test text had helped choose, its score would be tuned to it, not a fair measure. We touched it once, at the end, and we report whatever it says.
- **Why we compare with a counting baseline:** "far below untrained" is easy to meet. The bigram model asks a harder question: does the Transformer learn more than which word usually follows which? The size of the gap is the honest answer, even if it is small on the test patterns.
- **Why the extra checks were tested on real questions first:** a detector that also flags correct sentences would invent errors. On the real test questions the checks found nothing, so each flag on a generated sentence is a real error.
- **Why the units are learned, not hand-written:** the allowed unit for each item comes from the training questions themselves. That way the check matches the data, and nobody has to maintain a list by hand.
- **Why blind, separate scoring by two people:** if the scorers knew which method made an output, or talked to each other, their scores would not be independent. Hidden methods, shuffled order, separate copies and an agreement measure make the human scores trustworthy.
- **Why both agreement and kappa:** within 1 point is the guideline from task 6. Kappa removes the agreement expected by chance, so it shows whether the scorers truly see the same thing.
- **Why best.pt and last.pt can disagree:** validation loss measures how well the model predicts unseen text overall. Company-term correctness is a narrower property, and it can keep improving while the overall loss gets worse. The evaluation shows both, so the trade-off is visible instead of hidden.

## End of Day 13 checklist

- [ ] Files copied (`src\evaluation.py`, `scripts\day13_final_eval.py`, `scripts\day13_score_summary.py`, `prompts\day13_eval_prompts.csv`, `tests\test_evaluation.py`, both walkthroughs)
- [ ] `openpyxl` installed in `.venv`
- [ ] pytest: `81 passed`
- [ ] Part A finished; detectors on real test questions ≈ 0; `scoring_sheet.xlsx` written
- [ ] Copies sent; every scorer scored alone, before seeing the reports
- [ ] All copies in `docs\day13\scores\`; part B finished; `docs\DAY13_EVALUATION.md` exists
- [ ] D13 added with the real numbers; committed and pushed → **task 11 complete**

## Send back

1. The pytest result line.
2. The part A terminal output from `--- A1 perplexity` to the end.
3. The part B terminal output.
4. `docs\DAY13_AUTO_EVALUATION.md` and `docs\DAY13_EVALUATION.md`.
5. `docs\figures\day13_perplexity.png` and `docs\figures\day13_human_scores.png`.

If the scoring takes a few days because teammates need time, send part A first (items 1, 2 and the auto report). I'll check it while the scoring is going on.

## Report snippet (Japanese, for `docs/weekly_reports.md`)

```
【課題11】LLMの学習、動作確認、生成結果の評価(完了)
状況: 予定通り終わった(今週の予定: 最終の評価を行い、評価の表を作る)
内容: 作業6で決めた評価の方法と基準(変更なし)で、最終のモデルを評価した。
      テスト用のデータを初めて使い、パープレキシティを、学習前のモデルと、単純に数える方法
      (直前の1語だけから次を予測する方法)と比べた。固定の50の書き出しと2つの生成方法で、
      完結率・繰り返し率・distinct-2 などを測り、Day 12 で見つかった誤り(項目と単位が合わない、
      壊れた数字)を数える自動の確認も加えた(本物のテスト用の質問では誤検出0件)。
      最終のモデルの出力100件を、生成方法を隠して順番をばらばらにし、2人以上が別々に
      1〜5点で採点し、採点者の一致率も計算した。合格の目安と比べた最終の評価の表を作った。
      自動テスト7項目を追加し、合計81項目すべて合格。課題11はこれで完了。
証拠: docs/DAY13_AUTO_EVALUATION.md、docs/DAY13_EVALUATION.md、docs/figures/day13_*.png
次: 評価で見つかった問題(特に項目と単位の組み合わせ)をもとに、改善を考える
```
