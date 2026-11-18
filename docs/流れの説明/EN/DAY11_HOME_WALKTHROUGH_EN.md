# Day 11 (home laptop): size trials and the size decision

**Machine:** home laptop (RTX 4060 Laptop). **Repository:** `JC21-240447/llm_home`. **Folder:** `C:\Users\Merrick\Desktop\llm_home`.
**School plan:** **starts task 11** 「LLMの学習、動作確認、生成結果の評価」 (training, checking, evaluating). Task 11 runs over Days 11–13:
- Day 11 (today): choose the size;
- Day 12: the main training run;
- Day 13: the evaluation table.

Task 11 is complete after Day 13.

## What Day 11 does, and why

Day 8 prepared three sizes: tiny (about 1.0M parameters), small (5.2M) and medium (14.8M). Today we train all of them under **identical conditions** and choose one by **evidence**, using a rule written down **before** we see the results.

Day 9 changed how the trials must be run. In your `smoke_c` run, the validation loss was lowest at **step 200**, less than one pass over the training text, and rose after that. So:

1. **Validation every 25 steps, not 100.** The best point comes early and could otherwise be missed between two checks.
2. **Dropout as a second factor.** Dropout randomly switches off 10% of the signals during training, so the model cannot rely on memorising exact sequences. It is the standard remedy for the overfitting Day 9 found. Each size is trained with dropout 0.0 and 0.1, so 3 × 2 = **6 runs**.
3. **Generation is checked as well.** Day 9/10 showed that the lowest validation loss does not always write the best sentences. So each run's best checkpoint is also run on the 12 fixed Day 10 prompts, with the task 6 automatic checks.

**Identical conditions for all 6 runs:**
- batch 32 × 256 tokens, 800 steps (about 3 passes over the training text);
- warm-up 60, peak learning rate 0.001, seed 42.

Only the size and the dropout differ.

**The decision rule, written before the trials** (`src\selection.py`):
1. A run is a candidate only if its best checkpoint passes the task 6 guidelines on the 12 prompts (greedy + limits): **finished ≥ 90%** and **repetitive ≤ 5%**.
2. Among candidates, find the lowest validation loss.
3. Every candidate within **0.02** of it counts as equally good.
4. Among those, choose the one with the **fewest parameters**. If still tied, choose the lower loss, then no dropout.

Why prefer smaller within 0.02: a loss difference of 0.02 is about 2% in perplexity, which is run-to-run noise rather than evidence. A smaller model is faster, needs less memory, and memorises less.

New files:

| File | What |
| --- | --- |
| `src\selection.py` | summarises a run's `log.csv` (best point, gap, rise after best, speed, memory) and applies the decision rule |
| `scripts\day11_size_trials.py` | trains the 6 runs, checks generation on each `best.pt`, decides, writes the report, figure and decision file |
| `tests\test_selection.py` | 6 new tests for the summary and the rule |

Outputs:

| Output | What |
| --- | --- |
| `runs\day11\<size>_d<dropout>\` | one run folder per trial, e.g. `tiny_d00`, `medium_d10` (git-ignored) |
| `docs\DAY11_SIZE_TRIALS.md` | the decision, the results table, sample sentences |
| `docs\figures\day11_val_loss.png` | validation loss of all 6 runs in one figure |
| `configs\day11_decision.json` | the chosen size and dropout; **Day 12 reads this** |

**The script can be resumed.** It may be interrupted by Ctrl+C, the laptop sleeping, or a closed window. Run **the same command again**: finished runs are skipped and an unfinished run continues from its `last.pt`. I tested this with a hard kill in the middle of a run.

### SERVER NOTE (later)

Same code: `.venv/bin/python scripts/day11_size_trials.py --machine server`. On a server, start it inside `tmux` so it keeps running if you disconnect.

---

## Step 1. Put the files in place

1. Download `day11.zip` and unzip it.
2. Copy **everything inside** the `day11` folder into `llm_home`.
3. Check (VS Code terminal, `llm_home`):
   ```powershell
   dir src\selection.py, scripts\day11_size_trials.py, tests\test_selection.py
   ```

## Step 2. Run all tests (VS Code terminal, `llm_home`)

```powershell
.venv\Scripts\python.exe -m pytest tests -q
```

Expected: **`72 passed`** (66 + 6 new).

| No. | New test | What it proves |
| --- | --- | --- |
| 67 | `test_summary_reads_best_point_gap_and_rise` | from a log, it finds the lowest validation loss and its step, the training loss there, the gap, the final loss, how much it rose, the median speed and the peak memory |
| 68 | `test_lowest_loss_wins_when_clearly_better` | a clearly better (lower) loss wins, even for a bigger model |
| 69 | `test_smaller_model_wins_within_tolerance` | 0.015 worse but 5 times smaller: the smaller model is chosen |
| 70 | `test_runs_failing_the_task6_checks_are_never_chosen` | a run that does not finish 90% or repeats too much is never chosen, even with the lowest loss |
| 71 | `test_no_candidate_gives_no_choice` | if nothing passes the checks, no model is chosen (no silent guess) |
| 72 | `test_same_size_tie_prefers_lower_loss_then_no_dropout` | same size and same loss: no dropout is preferred |

## Step 3. Prepare the laptop

The trials take about **5–10 minutes**. Before starting:
- plug in the power adapter;
- in Windows Settings → System → Power, set "When plugged in, put my device to sleep after" to **Never**, at least for today;
- close heavy programs such as games or video editing that use the GPU.

## Step 4. Run the trials (VS Code terminal, `llm_home`)

```powershell
.venv\Scripts\python.exe scripts\day11_size_trials.py --machine home
```

What you will see:
1. `[1/6] tiny_d00: new`, then the training lines (every 25 steps), and so on until `[6/6] medium_d10`.
2. `--- generation check on each best.pt ---`, then one line per run.
3. The results table and the line **`Decision: <run> - <reason>`**.

If it stops for any reason, run the same command again.

## Step 5. Read the report

Open `docs\DAY11_SIZE_TRIALS.md` and press **Ctrl+Shift+V**. Look at:

1. **The figure.** One line per run: solid lines without dropout, dashed with dropout. Each dot is the lowest point of a line. Lines that go down and then up are overfitting.
2. **Best val loss and At step.** Which size reaches the lowest point, and how early.
3. **Gap** (validation minus training loss at the best step). A large gap means the model is much better at familiar sentence patterns than unseen ones.
4. **Rise after best.** How much the run got worse after its best point. Dropout should make this smaller.
5. **Finished / Repetitive / distinct-2** on each best checkpoint, and the **sample sentences**. Check by eye, for example whether 「勤続年数が」 is followed by years (correct) or yen (wrong).

I cannot predict which size wins. That is the point of running the trials. Day 9 suggests that all sizes will reach their best early and then rise, with the larger models rising sooner. Whatever the result, the rule decides, and the report shows why.

## Step 6. Record and commit

Add to `docs\DECISIONS.md`, filling in the numbers from your report:

```markdown
## D11. Model size (Day 11)

- Trials: tiny / small / medium x dropout 0.0 / 0.1, same budget (800 steps, batch 32 x 256, lr 1e-3, seed 42),
  validation every 25 steps; best checkpoints checked on the 12 Day 10 prompts.
- Rule (fixed before the trials): pass task 6 checks (finished >= 90 %, repetitive <= 5 %); within 0.02 of the
  lowest validation loss, choose the fewest parameters.
- Result: chosen <run> (<size>, dropout <d>), best validation loss <x.xxxx> at step <n>.
  Lowest overall: <run> <x.xxxx>. Observations: <e.g. larger models reached their best earlier and rose more;
  dropout reduced the rise after best by ...>.
- Day 12 trains the chosen configuration (configs/day11_decision.json).
```

Then commit (VS Code terminal, `llm_home`):

```powershell
git status --short
git add src scripts tests configs docs run_log.md
git status --short
git commit -m "Day 11 home: size trials and size decision"
git push
```

`configs\day11_decision.json` goes into Git. Nothing under `runs/` and no `.pt` file may appear in the list.

## Talking points for the review

- **Data versus parameters:** a model learns the shape of the data, but with too many parameters for the amount of text, it memorises the training questions instead. Our questions come from sentence patterns, so memorising them makes the model worse at unseen patterns, which is exactly what the validation loss measures.
- **Why not "bigger is better":** for large general-purpose models with huge amounts of text, bigger often wins. Here the amount of text is fixed and small, so the right size has to be measured, not assumed.
- **Why the rule was written first:** if we chose after seeing the numbers, we could always find a reason for any choice. A rule fixed in advance (checks, tolerance, prefer smaller) makes the decision honest and repeatable.
- **Why a tolerance of 0.02:** differences smaller than this are about 2% in perplexity, the size of run-to-run noise. Choosing the bigger model for a difference inside the noise would cost speed and memory for no proven gain.
- **Why dropout:** switching off random signals during training stops the model from relying on exact memorised sequences. It is the standard tool against overfitting, and the trials measure whether it helps here.
- **Why validate every 25 steps:** Day 9 found the best point within the first few hundred steps, so with checks every 100 steps the true best could fall between two checks.
- **Why also check sentences:** Day 9/10 showed that the lowest validation loss does not guarantee the best sentences, so a run must also pass the task 6 automatic checks to be chosen.

## End of Day 11 checklist

- [ ] Files copied (`src\selection.py`, `scripts\day11_size_trials.py`, `tests\test_selection.py`, both walkthroughs)
- [ ] pytest: `72 passed`
- [ ] Laptop on power, sleep off
- [ ] All 6 trials finished; `Decision:` line printed
- [ ] `docs\DAY11_SIZE_TRIALS.md`, `docs\figures\day11_val_loss.png`, `configs\day11_decision.json` exist
- [ ] D11 added with the real numbers; committed and pushed

## Send back

1. The pytest result line.
2. Everything from `--- generation check on each best.pt ---` to the end of the output (the checks, the table and the decision).
3. `docs\DAY11_SIZE_TRIALS.md` (the whole file, or as an attachment).
4. `docs\figures\day11_val_loss.png`.

With these I'll check the decision and prepare Day 12.

## Report snippet (Japanese, for `docs/weekly_reports.md`)

```
【課題11】LLMの学習、動作確認、生成結果の評価(開始)
状況: 予定通り終わった(今週の予定: 課題11の開始。モデルの大きさを決める)
内容: 3つの大きさのモデル(約100万・500万・1,500万パラメータ)を、ドロップアウトあり・なしで、
      同じ条件で学習させて比べた(6回)。前もって決めたルール(作業6の自動の確認に合格し、
      検証の損失が一番低いものとの差が0.02以内なら、パラメータが少ない方を選ぶ)で、
      本番のモデルの大きさを決めた。自動テスト6項目を追加し、合計72項目すべて合格。
証拠: docs/DAY11_SIZE_TRIALS.md、docs/figures/day11_val_loss.png、configs/day11_decision.json
次: 決めた大きさで本番の学習を行う(Day 12)
```
