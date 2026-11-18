# Day 12 (home laptop): the main training run

**Machine:** home laptop (RTX 4060 Laptop). **Repository:** `JC21-240447/llm_home`. **Folder:** `C:\Users\Merrick\Desktop\llm_home`.
**School plan:** task 11 「LLMの学習、動作確認、生成結果の評価」 **continues** (Day 11 = choose the size, **Day 12 = main training**, Day 13 = evaluation table). Task 11 is complete after Day 13.

## What Day 12 does, and why

Day 11 chose **tiny with dropout 0.1** (`configs\day11_decision.json`). Today that configuration is trained as the **main model**, the one Day 13 evaluates. Three things change from the Day 11 trials, each because of a Day 11 finding:

1. **Three seeds instead of one (42, 43, 44).** Your seed-43 check showed that the same configuration can differ by up to **0.062** in validation loss just from the seed. One number is therefore not a reliable result, so we report the **mean and the spread** of three runs. The final model is the seed with the lowest validation loss.
2. **A shorter schedule: 400 steps instead of 800.** In Day 11 the best point came at steps 125–225, while the learning rate was still high (it only decays toward the end of the schedule). With 400 steps, the learning rate is already decreasing when the best point arrives, so the model takes smaller, more careful steps there.
3. **Early stopping.** Validation every 10 steps; if the validation loss does not improve for **100 steps**, training ends. Training past the best point only makes the model memorise more, and `best.pt` already keeps the best point, so continuing wastes time.

**The test set is not touched today.** We choose the final model using only the validation set. The test set is kept for Day 13, so that the final evaluation is done on text the choices never saw.

Conditions for all 3 runs:
- tiny, dropout 0.1 (read from `configs\day11_decision.json`, not typed again);
- batch 32 × 256 tokens, 400 steps, warm-up 60, peak learning rate 0.001, cosine down to 10%;
- validation every 10 steps, early stopping patience 100.

Only the seed differs.

### Changed files

| File | Change |
| --- | --- |
| `src\training.py` | **early stopping** (`early_stop_patience`); the best step is saved in the checkpoint so it survives Ctrl+C and resume; at the end each run writes `status.json` (`finished`, `early_stopped` or `stopped`) |
| `src\generation.py` | `quick_checks()`: the task 6 automatic checks (finished, repetitive, distinct-2, new sentences) on the fixed prompts, as one function |
| `tests\test_training.py` | 2 new tests for early stopping |

### New files

| File | What |
| --- | --- |
| `scripts\day12_train_main.py` | trains the 3 seeds, checks generation on each `best.pt`, chooses the final model, compares with Day 11, writes the report, figure and final-model file |
| `scripts\day12_monitor.py` | shows a run's progress, best point, early-stopping counter and time left (optional; from a second terminal) |

### Outputs

| Output | What |
| --- | --- |
| `runs\day12\main_s42\`, `main_s43\`, `main_s44\` | one run folder per seed (git-ignored) |
| `docs\DAY12_MAIN_TRAINING.md` | the final model, all seeds, comparison with Day 11, sample sentences |
| `docs\figures\day12_main.png` | training (dashed) and validation (solid) loss of the 3 seeds |
| `configs\day12_final.json` | which checkpoint is the final model; **Day 13 reads this** |

**The script can be resumed**, like Day 11. If it is interrupted, run **the same command again**: finished or early-stopped seeds are skipped, and an unfinished seed continues from its `last.pt`, with its early-stopping counter intact. I tested this by killing the script in the middle of the third seed.

### SERVER NOTE (later)

Same code: `.venv/bin/python scripts/day12_train_main.py --machine server`, started inside `tmux`.

---

## Step 1. Put the files in place

1. Download `day12.zip` and unzip it.
2. Copy **everything inside** the `day12` folder into `llm_home`. Replace `src\training.py`, `src\generation.py` and `tests\test_training.py` when Windows asks.
3. Check (VS Code terminal, `llm_home`):
   ```powershell
   dir scripts\day12_train_main.py, scripts\day12_monitor.py
   ```

## Step 2. Run all tests (VS Code terminal, `llm_home`)

```powershell
.venv\Scripts\python.exe -m pytest tests -q
```

Expected: **`74 passed`** (72 + 2 new).

| No. | New test | What it proves |
| --- | --- | --- |
| 73 | `test_early_stopping_ends_the_run_and_keeps_the_best` | on data where validation stops improving, the run ends before `max_steps`, `status.json` says `early_stopped`, and `best.pt` holds the best point (not the last one) |
| 74 | `test_early_stopping_survives_stop_and_resume` | a run stopped with Ctrl+C and resumed stops at exactly the same step, with the same best point, as a run that was never interrupted |

The existing tests (including the "resume gives identical results" tests from Day 9) still pass, so adding early stopping changed nothing else in training.

## Step 3. Prepare the laptop

The 3 runs take about **3–6 minutes** in total (tiny is fast; validation every 10 steps adds a little). As on Day 11: power adapter in, sleep set to **Never**, heavy GPU programs closed.

## Step 4. Run the main training (VS Code terminal, `llm_home`)

```powershell
.venv\Scripts\python.exe scripts\day12_train_main.py --machine home
```

What you will see:
1. `=== Day 12 main training | tiny, dropout 0.1 (from day11_decision.json) | seeds [42, 43, 44] | ... ===`. Check that it says **tiny, dropout 0.1**.
2. `[1/3] main_s42: new`, then training lines every 10 steps. A line marked `* best, saved best.pt` is a new best point.
3. Probably, a line like `Early stop at step 3xx: no improvement for 100 steps (best validation loss ... at step ...; best.pt keeps it).` This is expected and good, not an error.
4. The same for `main_s43` and `main_s44`.
5. `--- generation check on each best.pt ---`, then the summary: the loss per seed, the mean and spread, the comparison with Day 11, and **`Final model: main_sXX`**.

If it stops for any reason, run the same command again.

**Optional: watch a run live.** Open a second terminal (the **+** in the VS Code terminal panel) in `llm_home` and run:

```powershell
.venv\Scripts\python.exe scripts\day12_monitor.py --run runs/day12/main_s42 --watch 5
```

It refreshes every 5 seconds and shows the progress bar, the best point, `early stop N / 100 steps without improvement`, and the time left. It only reads the log, so it cannot disturb the training. Press **Ctrl+C** in that terminal to quit it (Ctrl+C there does not stop the training). After the training, the same command without `--watch 5` prints the final state once.

## Step 5. Read the report

Open `docs\DAY12_MAIN_TRAINING.md` and press **Ctrl+Shift+V**. Look at:

1. **The figure.** Three colours = three seeds. Solid = validation, dashed = training. The dots are the `best.pt` points. The lines should lie close together; where they separate shows the seed noise. Where the solid line flattens and the dashed line keeps falling, the model is starting to memorise. That is where early stopping ended the run.
2. **All seeds table.** *How it ended* (`early_stopped at step ...` or `finished at step 400`), *Best val loss*, *At step*, and *Gap* (validation minus training at the best step). Then the line **Mean ... spread ...**.
3. **Compared with the Day 11 trials.** Day 11 tiny_d10 averaged 2.6637 over 2 seeds with the 800-step schedule. **Read the change together with the spread:** if the change is smaller than the spread between seeds, the shorter schedule made no proven difference. That is a valid result, and we report it as it is.
4. **Sample sentences** from the final model, and the Finished / Repetitive columns: all three should pass the task 6 guidelines (finished ≥ 90%, repetitive ≤ 5%).

I cannot predict the exact numbers. My expectation is a mean close to Day 11 (about 2.64–2.70) with a spread around 0.05, since Day 11 showed that the data, not the settings, limits the result. Whatever comes out, we record it honestly.

## Step 6. Record and commit

Add to `docs\DECISIONS.md`, filling in the numbers from your report:

```markdown
## D12. Main model (Day 12)

- Configuration from D11 (tiny, dropout 0.1), 400-step schedule (warm-up 60, lr 1e-3, cosine to 10 %),
  validation every 10 steps, early stopping after 100 steps without improvement, seeds 42 / 43 / 44.
- Best validation loss per seed: <x.xxxx> / <x.xxxx> / <x.xxxx> (mean <x.xxxx>, spread <x.xxxx>).
  Each run ended: <early_stopped at step n / finished>.
- Compared with the Day 11 trials (800 steps, mean 2.6637): change <+/-x.xxxx>, <inside / outside> the seed spread.
- Final model: runs/day12/main_s<seed>/best.pt (lowest validation loss; passes the task 6 checks).
  Recorded in configs/day12_final.json. The test set was not used; it is kept for Day 13.
```

Then commit (VS Code terminal, `llm_home`):

```powershell
git status --short
git add src scripts tests configs docs run_log.md
git status --short
git commit -m "Day 12 home: main training with early stopping, 3 seeds, final model"
git push
```

`configs\day12_final.json` goes into Git (it is small; the `.pt` file it points to is not). Nothing under `runs/` and no `.pt` file may appear in the list.

**Keep the run folders.** `runs\day12\main_s<seed>\best.pt` is the final model, and Day 13 needs it. Do not delete `runs\day12\`.

## Talking points for the review

- **Why three seeds:** the seed decides the starting weights and the order of the training windows. Day 11 measured that this alone changes the validation loss by up to 0.06. Reporting one run would hide that noise, so we report the mean and the spread, which shows how much of any difference is real.
- **Why early stopping:** after the best point the model only memorises the training questions, and the validation loss rises. Stopping after 100 steps without improvement saves time, and the model we keep is still the best point, saved in `best.pt`.
- **Why 100 steps of patience:** the validation loss goes up and down a little from check to check. Stopping at the first rise would stop too early by chance. 100 steps is 10 checks, long enough to be sure the best point has really passed.
- **Why a shorter schedule:** the learning rate falls along a cosine curve over the whole schedule. With 800 steps it was still high at the best point (steps 125–225); with 400 steps it is already lower there, so the model settles more gently.
- **Why the test set is not used today:** every choice (size, dropout, seed) is made with the validation set. If the test set were also used for choosing, its score would no longer be an honest measure of unseen text. Day 13 uses it once, at the end.
- **Why the result may look the same as Day 11:** Day 11 showed that the amount and kind of training text, not the settings, sets the limit. A main run that confirms this with three seeds is still the evidence we need.

## End of Day 12 checklist

- [ ] Files copied (`src\training.py`, `src\generation.py`, `tests\test_training.py`, `scripts\day12_train_main.py`, `scripts\day12_monitor.py`, both walkthroughs)
- [ ] pytest: `74 passed`
- [ ] Header says `tiny, dropout 0.1`
- [ ] All 3 seeds ended (early stopped or finished); `Final model:` line printed
- [ ] `docs\DAY12_MAIN_TRAINING.md`, `docs\figures\day12_main.png`, `configs\day12_final.json` exist
- [ ] D12 added with the real numbers; committed and pushed; `runs\day12\` kept

## Send back

1. The pytest result line.
2. Everything from `--- generation check on each best.pt ---` to the end of the output.
3. `docs\DAY12_MAIN_TRAINING.md` (the whole file, or as an attachment).
4. `docs\figures\day12_main.png`.
5. `configs\day12_final.json`.

With these I'll check the final model and prepare Day 13, the last day of task 11.

## Report snippet (Japanese, for `docs/weekly_reports.md`)

```
【課題11】LLMの学習、動作確認、生成結果の評価(続き)
状況: 予定通り終わった(今週の予定: 決めた大きさで本番の学習を行う)
内容: Day 11 で決めた設定(tiny、ドロップアウト0.1)で本番の学習を行った。乱数の種を3つ
      (42・43・44)使って、結果を平均とばらつきで示した。検証の損失が100ステップ改善しなければ
      学習を止める「早期終了」を学習の仕組みに加え、学習の予定を800ステップから400ステップに
      短くした。検証の損失が一番低い種のモデルを最終のモデルとした。テスト用のデータは
      最後の評価のために使っていない。自動テスト2項目を追加し、合計74項目すべて合格。
証拠: docs/DAY12_MAIN_TRAINING.md、docs/figures/day12_main.png、configs/day12_final.json
次: テスト用のデータで最終の評価を行い、評価の表を作る(Day 13、課題11の完了)
```
