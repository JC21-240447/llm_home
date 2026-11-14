# Day 9 (home laptop): the trainer, a first small training run, and stop/resume

**Machine:** home laptop (RTX 4060 Laptop). **Repository:** `JC21-240447/llm_home`. **Folder:** `C:\Users\Merrick\Desktop\llm_home`.
**School plan:** continues **task 10** (the next-word mechanism). Today's loss curve is the evidence that the mechanism can learn. Task 10 is finished on Day 10 with the top-10 next-word candidates.

## What Day 9 does, and why

Day 8 built the model and proved it is wired correctly. Today it **learns for the first time**. We build the trainer that Days 11 and 12 will also use, and we prove that it can be stopped and resumed without changing the result.

What the trainer does at every step:

1. Takes a batch of 32 **windows** of 256 tokens from random positions in `train.bin`. In each window, the model reads tokens 1–256 and must predict tokens 2–257, so one window gives 256 "next token" examples.
2. Computes the loss in **bf16** (half-size numbers the 4060 handles natively), then the gradients.
3. Limits the gradient size (**clipping**), then updates the weights with **AdamW**.
4. Sets the **learning rate**: it rises for the first 60 steps (warm-up), then falls along a cosine curve.

Every 10 steps it writes a row to `log.csv`. Every 100 steps it measures the **validation loss** on the whole validation set, writes greedy continuations of four fixed prompts to `samples.md`, and saves a **checkpoint**.

New and changed files:

| File | What |
| --- | --- |
| `src\batching.py` | random training windows (with a saveable random state) and fixed validation windows, read from disk with a memory map |
| `src\training.py` | the training loop: learning-rate schedule, AdamW with weight-decay groups, clipping, gradient accumulation, bf16/fp16, validation, samples, checkpoints, resume, safe Ctrl+C |
| `scripts\day9_train.py` | command line for a new run or `--resume` |
| `scripts\day9_compare_runs.py` | compares a non-stop run with a stopped-and-resumed run, and draws the loss figure and report |
| `tests\test_training.py` | 8 new tests, including "stopped + resumed = uninterrupted, to the last digit" |
| `requirements.txt` | adds `matplotlib` (for the figure) |

A run is saved in `runs\day9\<run name>\`, which Git ignores:

| File | What |
| --- | --- |
| `train_config.json`, `model_config.json` | the settings of the run |
| `log.csv` | step, loss, perplexity, learning rate, gradient size, tokens/s, GPU memory, time |
| `samples.md` | at every evaluation, what the model writes after 「営業部の」「ボーナスが」「大阪支社で」「社員の平均」 |
| `last.pt` | the latest checkpoint (used for resuming) |
| `best.pt` | the checkpoint with the lowest validation loss so far |

### How resuming stays exact

`last.pt` holds the weights, the optimizer state (AdamW's running averages), the step number, the random number generators, and the state of the generator that picks the training windows. A resumed run therefore gets the same windows and the same learning rates as a run that never stopped. A test proves the losses are identical to the last digit on the CPU.

Two safety details:
- Checkpoints are written to a temporary file first and then renamed. An interruption while saving never leaves a broken `last.pt`.
- If the program is killed (the window closed, the power lost), `log.csv` may have rows after the last checkpoint. On resume they are removed and trained again, so the log stays clean.

### SERVER NOTE (later)

The same code runs on Linux: `.venv/bin/python scripts/day9_train.py --machine server --run-name ...`. On a GPU without bf16 (an RTX 2060), `--precision auto` chooses fp16 with a gradient scaler by itself. On a server you may want to start long runs inside `tmux` so they keep running after you disconnect. I'll explain that when we set up the server.

---

## Step 1. Put the files in place and install matplotlib

1. Download `day9.zip` and unzip it.
2. Copy **everything inside** the `day9` folder into `llm_home`. When Windows asks, **replace** `requirements.txt` (「ファイルを置き換える」 / "Replace the file in the destination"). The new file only adds one line at the end.
3. Install matplotlib **into the venv** (VS Code terminal, `llm_home`):
   ```powershell
   .venv\Scripts\python.exe -m pip install "matplotlib>=3.8"
   .venv\Scripts\python.exe -c "import matplotlib; print(matplotlib.__version__)"
   ```
   The second line prints a version, for example 3.11.x.
4. Check:
   ```powershell
   dir src, tests, scripts\day9_*.py
   ```
   `src` must show `batching.py` and `training.py` as well, and `tests` must show four test files.

## Step 2. Run all tests (VS Code terminal, `llm_home`)

```powershell
.venv\Scripts\python.exe -m pytest tests -q
```

Expected: **`54 passed`** (46 + 8 new). The new tests train a very small model on the CPU for a few steps, so they take about 10 seconds.

| New test | What it proves |
| --- | --- |
| `test_lr_schedule_warmup_then_cosine` | warm-up reaches the peak, then the rate only falls, ending at 10% of the peak |
| `test_weight_decay_only_on_matrices` | weight decay is applied to the matrices but not the norm gains, and every parameter is in exactly one group |
| `test_random_windows_shift_and_resume_state` | the target is the input shifted by one token, and a saved random state gives the same next batch |
| `test_validation_windows_cover_the_stream_without_overlap` | validation measures every token once, the same way every time |
| `test_stopped_and_resumed_run_equals_uninterrupted_run` | **stop at step 6, resume, finish: every loss equals the non-stop run** |
| `test_resume_refuses_changed_settings` | a resumed run with different settings is refused (it would no longer be the same run) |
| `test_new_run_refuses_an_existing_folder` | a new run never overwrites an old one by accident |
| `test_resume_after_a_hard_kill_…` | rows left behind by a killed program are removed on resume |

## Step 3. The first training run, without stopping (VS Code terminal, `llm_home`)

```powershell
.venv\Scripts\python.exe scripts\day9_train.py --run-name smoke_a
```

Default settings: the tiny model, batch 32 × 256 tokens, 600 steps, warm-up 60, peak learning rate 0.001. That is about 4.9 million tokens, about 2.4 passes over the training text. On your GPU it should take **under a minute**.

What to watch:
- **step 0:** validation loss about **7.4**. This is the untrained model guessing evenly, as on Day 8.
- The training loss falls quickly in the first 100–200 steps, then slowly.
- The validation loss falls too, but stays **higher** than the training loss. This is expected: validation uses sentence patterns the model never sees in training (Day 5). The size of this gap is what Day 11 studies.

I ran a shorter version on my CPU (batch 8, 300 steps, a quarter of the tokens). Validation went from 7.45 to about 2.72, and training loss reached about 1.28. Your run sees more text, so your numbers will be somewhat lower. Send me yours.

Then look at `runs\day9\smoke_a\samples.md`. At step 0 the model repeats one piece (「ののののの…」). At the end it writes real question endings, for example in my CPU run:

```text
| 大阪支社で | 働く社員の月収の中央値を出してください。 ⟨eos⟩ |
| 営業部の   | 社員を教えてください。 ⟨eos⟩ |
| ボーナスが | 3000000000000000000000000000000000000000 |
```

The last line is a typical weakness of **greedy** generation, where the model always takes the single top token: once it is in a loop, it never leaves. Day 10 fixes this with sampling and repetition control.

## Step 4. The same run, stopped halfway and resumed (VS Code terminal, `llm_home`)

```powershell
.venv\Scripts\python.exe scripts\day9_train.py --run-name smoke_b --stop-after 300
.venv\Scripts\python.exe scripts\day9_train.py --run-name smoke_b --resume
```

The first command stops at step 300 and prints `Stopped at step 300. Continue with --resume.` The second prints `Resumed from last.pt at step 300` and continues to 600. With `--resume`, all settings are read from the run folder; you don't repeat them.

## Step 5. Compare the two runs (VS Code terminal, `llm_home`)

```powershell
.venv\Scripts\python.exe scripts\day9_compare_runs.py --a smoke_a --b smoke_b
```

Expected:

```text
Final step: smoke_a 600, smoke_b 600 (resumed at 300)
  same final step              PASS
  training losses match        PASS
  validation losses match      PASS
  no jump at the resume step   PASS
Largest difference between the runs: ... -> identical   (or: the same curve)
```

On my CPU the difference was exactly 0 ("identical"). On a GPU a few calculations, such as parts of the fast attention's backward pass, may add numbers in a slightly different order from run to run. So the GPU difference can be a tiny non-zero number, reported as "the same curve". Both mean resuming works. If you see FAIL or DIFFERENT, send me the output.

Open `docs\DAY9_SMOKE_REPORT.md` in VS Code and press **Ctrl+Shift+V** for the preview. It shows the figure `docs\figures\day9_smoke_loss.png`: the two training curves lie on top of each other, including across the dotted line at step 300. **This figure is the main evidence for task 10 today.**

## Step 6 (optional, 1 minute). Stop with Ctrl+C yourself

This shows the real-life case: you need the laptop while it trains.

```powershell
.venv\Scripts\python.exe scripts\day9_train.py --run-name smoke_c --max-steps 3000
```

After a few seconds, press **Ctrl+C once**. It finishes the current step, saves `last.pt` and stops. Then:

```powershell
.venv\Scripts\python.exe scripts\day9_train.py --run-name smoke_c --resume
```

It continues from where you stopped. Press Ctrl+C **twice** only if you need it to stop at once; it then continues later from the last saved checkpoint.

## Step 7. Record and commit

Add to `docs\DECISIONS.md`:

```markdown
## D9. Trainer (Day 9)

- Training examples: random 256-token windows from the train stream (x = window, y = window shifted by one).
  Validation: fixed non-overlapping windows covering the whole validation stream.
- AdamW (betas 0.9/0.95, weight decay 0.1 on matrices only), linear warm-up then cosine decay to 10% of the peak,
  gradient clipping at 1.0, bf16 autocast on the RTX 4060 (fp16 + gradient scaler on GPUs without bf16).
- Checkpoints hold the model, optimizer, scaler, step and all random states; a stopped and resumed run gives the
  same losses as an uninterrupted run (test on CPU: identical; smoke run on GPU: see docs/DAY9_SMOKE_REPORT.md).
- Smoke run settings: tiny model, batch 32 x 256, 600 steps, warm-up 60, peak lr 1e-3.
```

Then commit (VS Code terminal, `llm_home`). The `runs` folder is ignored by Git; the report and the figure go in:

```powershell
git status --short
git add requirements.txt src scripts tests docs run_log.md
git status --short
git commit -m "Day 9 home: trainer, smoke run, stop/resume check"
git push
```

No `.pt` file and nothing under `runs/` may appear in the list.

## Talking points for the review

- **What the loss is:** for each position, the model gives a probability to the real next token. The loss is the average of −log(that probability). If it gives the right token probability 1, the loss is 0. A loss of 7.42 means it spread its guess evenly over all 1,674 tokens.
- **What perplexity is:** e^loss, "how many tokens the model is still choosing between, on average". It starts at about 1,674, and in my CPU test it ended near 15 on validation. Lower means more certain and more often right.
- **Why warm-up:** at the start the weights are random and the gradients are large and noisy. Starting with a small learning rate and raising it over 60 steps avoids wild first updates. The cosine decay at the end lets the model settle into a good point.
- **Why AdamW and why no weight decay on the norm gains:** AdamW adapts the step size for each weight. Weight decay gently pulls the large matrices toward zero to reduce memorisation, but the norm gains are scaling factors that should stay near 1, so they are left out.
- **Why gradient clipping:** one unusual batch can produce a huge gradient and wreck the weights. Clipping caps the size of every update.
- **Why bf16:** half-size numbers make training faster and use less memory. bf16 keeps the same range as float32, so it needs no loss scaling. On the RTX 2060, which has no bf16, fp16 is used with a gradient scaler.
- **Why the validation loss is higher than the training loss:** validation holds out whole sentence patterns (Day 5), so it measures how well the model handles wording it has never seen, not how well it remembers. That is the honest measure, and Day 11 uses it to choose the model size.
- **Why resuming must be exact:** long runs (Day 12) will be interrupted. If resuming changed the batches or the learning rate, an interrupted run would not be the same experiment. Saving every random state makes it the same, and the test and the figure prove it.

## End of Day 9 checklist

- [ ] Files copied, `requirements.txt` replaced, matplotlib installed in `.venv`
- [ ] pytest: `54 passed`
- [ ] `smoke_a` finished 600 steps; validation loss started at about 7.4 and fell
- [ ] `smoke_b` stopped at 300 and resumed to 600
- [ ] Compare: four PASS; verdict "identical" or "the same curve"
- [ ] `docs\DAY9_SMOKE_REPORT.md` and `docs\figures\day9_smoke_loss.png` exist
- [ ] (optional) Ctrl+C stop and resume tried
- [ ] D9 added; committed and pushed; no `.pt` files in Git

## Send back

1. The pytest result line.
2. The last 10 lines of Step 3 (the final losses).
3. The full output of Step 5 (the comparison).
4. The last section of `runs\day9\smoke_a\samples.md` (the step 600 samples).

## Report snippet (Japanese, for `docs/weekly_reports.md`)

```
【課題10】次に続く単語を予測する仕組み(途中)
状況: 予定通り終わった(今週の予定: 学習プログラムの作成と小さな学習)
内容: 学習プログラムを作成し、自宅ノートPCの GPU で小さなモデルの学習を行った。損失が学習前の約7.4から
      下がり、次の単語を予測できるようになっていくことを確認した。途中で止めて再開しても、止めなかった
      場合と同じ損失の曲線になることを確認した(自動テスト8項目を追加、合計54項目すべて合格)。
証拠: docs/DAY9_SMOKE_REPORT.md、docs/figures/day9_smoke_loss.png、run_log.md、tests/
次: 次の単語の候補トップ10を表示する仕組みを作り、課題10を完了する(Day 10)
```
