# Day 10 (home laptop): next-word candidates, sampling and repetition control

**Machine:** home laptop (RTX 4060 Laptop). **Repository:** `JC21-240447/llm_home`. **Folder:** `C:\Users\Merrick\Desktop\llm_home`.
**School plan:** **completes task 10** (the next-word mechanism). Its three pieces of evidence are now all there:
- the model code with passing tests (Day 8),
- the loss curve of the first training run (Day 9),
- the **top-10 next-word candidates table** (today).

## What Day 10 does, and why

Day 9 trained the model, and `samples.md` showed what it writes when it always takes the most likely token. Today we build the part that turns the model's probabilities into text properly. We also show the probabilities themselves, which is the school plan's task 10 goal: given 「今日」「は」, what does the model think comes next?

**How next-word probabilities become text, one token at a time:**

1. The model reads the tokens so far and gives every one of the 1,674 tokens a score. softmax turns the scores into probabilities that add up to 1. **The candidates view shows the top 10.**
2. The scores are adjusted:
   - special tokens (`<pad>`, `<bos>`, `<unk>`, `<unused_*>`) are removed;
   - repetition limits are applied (see below).
3. One token is chosen:
   - **greedy:** always the top token. The same input always gives the same output.
   - **sampling:** temperature, then top-k, then top-p, then a random draw with a fixed seed.
4. Repeat until `<eos>` (the end of the question) or 40 tokens.

**The repetition limits were measured from the training data.** Before choosing limits, I measured the training questions:
- the same token appears **at most 4 times in a row** (「社員番号100003」);
- a 3-token sequence appears **at most twice** in one question (「1991年から1997年」).

So the limits are exactly those numbers: a 5th identical token in a row, or a 3rd copy of the same 3-token sequence, is not allowed. This stops loops such as 「5000000…」 but can never block a real question.

A common default, "never repeat any 3-token sequence", would have blocked real questions like 「1991年から1997年に入社した社員」. Measuring first avoided that mistake.

New files:

| File | What |
| --- | --- |
| `src\generation.py` | candidates, score adjustment, greedy and sampling, `generate()`, the automatic checks from task 6 (finished, repetitive, distinct-2) |
| `scripts\day10_generate.py` | three commands: `candidates`, `generate`, `report` |
| `prompts\day10_prompts.txt` | 12 fixed beginnings: 9 company ones, 1 harder one (「社員の平均」), 2 out of scope (「今日は」「天気が」) |
| `tests\test_generation.py` | 12 new tests |

The four settings compared (fixed, so the report can be reproduced):

| Setting | What it is | Why it is in the comparison |
| --- | --- | --- |
| `greedy` | top token, no controls | what Day 9 `samples.md` shows |
| `greedy+limits` | greedy + repetition limits | the safe deterministic setting |
| `sample` | temperature 0.8, top-k 50, top-p 0.95, limits | varied but careful: the setting we expect to use |
| `sample-hot` | temperature 1.5, limits | too much randomness, to show what goes wrong |

**One fix made while building.** In my first version, every prompt restarted the random generator with the same seed 42, so different prompts got the same random draws. 「営業部の」「開発部の」「東京本社の」 then all produced the identical sampled sentence. Now each prompt gets its own fixed seed, made from the seed and the prompt. Results are still exactly the same on every run, and different prompts are independent. A test checks this.

### SERVER NOTE (later)

Same code: `.venv/bin/python scripts/day10_generate.py report --run smoke_a --machine server`. Generation is light, so `--device cpu` works on any PC.

---

## Step 1. Put the files in place

1. Download `day10.zip` and unzip it.
2. Copy **everything inside** the `day10` folder into `llm_home`. The `prompts` folder is new.
3. Check (VS Code terminal, `llm_home`):
   ```powershell
   dir src\generation.py, scripts\day10_generate.py, tests\test_generation.py, prompts
   ```

## Step 2. Run all tests (VS Code terminal, `llm_home`)

```powershell
.venv\Scripts\python.exe -m pytest tests -q
```

Expected: **`66 passed`** (54 + 12 new).

| New test | What it proves |
| --- | --- |
| `test_candidates_are_sorted_probabilities` | the top 10 are real softmax probabilities, largest first |
| `test_special_tokens_are_never_written_but_eos_is_allowed` | `<bos>`, `<pad>` and the like can never be written; `<eos>` can |
| `test_same_token_run_limit_matches_training_data` | 4 in a row is allowed (as in 社員番号100003); a 5th is not |
| `test_trigram_limit_matches_training_data` | a 3-token sequence may appear twice (as in 1991年から1997年), not three times |
| `test_repetition_penalty_lowers_only_used_tokens` | the optional penalty changes only tokens already used |
| `test_greedy_takes_the_top_token` | greedy = the highest score |
| `test_top_k_one_and_tiny_top_p_behave_like_greedy` | top-k and top-p remove the right tokens |
| `test_temperature_changes_how_often_the_second_token_is_chosen` | measured over 2,000 draws: at temperature 1 the second token is chosen as often as softmax says; lower temperature means less often, higher means more often |
| `test_prompt_seed_is_fixed_per_prompt_and_differs_between_prompts` | the seed fix described above |
| `test_generate_stops_at_eos_and_is_reproducible` | generation ends at `<eos>`, stops at the length limit, and gives the same result twice |
| `test_repetition_check_and_distinct_2` | the task 6 checks A3 and A4 are calculated correctly |
| `test_invalid_settings_are_refused` | impossible settings give a clear error |

## Step 3. Next-word candidates: the school plan example (VS Code terminal, `llm_home`)

```powershell
.venv\Scripts\python.exe scripts\day10_generate.py candidates --text 今日は --text 営業部の --out docs\DAY10_NEXT_TOKEN_CANDIDATES.md
```

It uses `runs\day9\smoke_a\best.pt` by default. I ran the same code on the checkpoints you sent:

```text
今日は  ->  pieces: <e4bb> / <8a> / 日 / は
   1. 誰             71.97%
   2. 何人             6.51%
   3. 教              2.14%
   ...
      top 10 total: 86.92%
```

On your GPU the last digits may differ slightly. What this shows:
- 「今」 is not in the training questions, so it is written as two byte pieces (as in the Day 7 demo).
- The model has learned only company questions, so after any 「…は」 it expects a question word: 「誰」 (who), 「何人」 (how many people).
- It does not know about weather, so it cannot continue 「今日は」 with 「晴れ」. **The candidates show exactly what the model has learned.**

## Step 4. The full Day 10 report (VS Code terminal, `llm_home`)

```powershell
.venv\Scripts\python.exe scripts\day10_generate.py report --run smoke_a --machine home
```

It runs both checkpoints of your Day 9 run:
- `best.pt`, step 200, the lowest validation loss;
- `last.pt`, step 600.

It writes `docs\DAY10_GENERATION_REPORT.md` with four sections:
1. the top-10 candidates for 4 beginnings, `best.pt` and `last.pt` side by side;
2. every prompt with all 4 settings on both checkpoints;
3. the automatic checks from task 6;
4. the repetition limits on the untrained (step 0) model.

It takes under a minute (45 seconds on my CPU). Expected automatic checks, from your checkpoints on my CPU:

```text
ckpt  setting         finished  repetitive  distinct-2    new
best  greedy             12/12        0/12        0.56  12/12
last  greedy             12/12        0/12        0.47   8/12
best  greedy+limits      12/12        0/12        0.56  12/12
last  greedy+limits      12/12        0/12        0.47   8/12
best  sample             12/12        0/12        0.76  11/12
last  sample             12/12        0/12        0.80   8/12
best  sample-hot         10/12        0/12        0.98  10/10
last  sample-hot         12/12        0/12        0.92   9/12
```

The greedy rows must match. They also match your Day 9 `samples.md`: 「大阪支社で」 gives 「働く社員の2ですか。」 from `best.pt` and 「働く社員の時間外手当の中央値を出してください。」 from `last.pt`. The sampling rows use random draws, so a GPU can change a word here and there.

## Step 5. Read the report: five findings

Open `docs\DAY10_GENERATION_REPORT.md` and press **Ctrl+Shift+V**.

1. **`last.pt` uses company terms better than `best.pt`.** With 「勤続年数が」 (years of service), `best.pt` writes 「30万円以上の社員…」 (¥300,000: money for a number of years, an H4 error in task 6). `last.pt` writes 「30年以下の社員を教えてください。」, which is correct. This confirms the Day 9 finding: the lowest validation loss is not the best writer for familiar beginnings.
2. **Sampling gives variety without breaking sentences.** `sample` finishes all 12 questions on both checkpoints, and distinct-2 rises from about 0.5 (greedy) to about 0.8. Example: 「ボーナスが」 → 「500万円以上の社員は何人いますか。」
3. **Too much randomness breaks the text.** `sample-hot` (temperature 1.5) writes broken characters (�) and word salad. Temperature flattens the probabilities, so rare byte pieces and unrelated words get chosen. That is why `sample` uses 0.8 with top-k and top-p.
4. **The repetition limits work, and they're a safety net.** The trained checkpoints no longer loop on these prompts, so `greedy` and `greedy+limits` are identical there. Section 4 rebuilds the untrained model with the Day 9 seed, which reproduces the 「のののの…」 from `samples.md` exactly. Greedy loops forever; with the limits it stops after 4 「の」 and moves on.
5. **Out-of-scope beginnings become company questions.** 「今日は」 becomes 「誰ですか。」 and 「天気が」 becomes 「入っている社員を教えてください。」. The model has only ever seen company questions, so it turns everything into one. Recognising out-of-scope input is not part of this model. The task 6 evaluation records it under the "out of scope" beginnings.

## Step 6 (optional). Try your own beginnings

```powershell
.venv\Scripts\python.exe scripts\day10_generate.py generate --setting sample --checkpoint runs\day9\smoke_a\last.pt --text 人事部で --text 年齢が
```

You can also change one setting, for example `--temperature 1.0` or `--seed 7`.

## Step 7. Record and commit

Add to `docs\DECISIONS.md`:

```markdown
## D10. Generation (Day 10)

- Repetition limits measured from the training questions: a token at most 4 times in a row (社員番号100003),
  a 3-token sequence at most twice (1991年から1997年). A generic "no repeated 3-gram" rule would block real questions.
- Settings compared: greedy, greedy+limits, sample (T 0.8, top-k 50, top-p 0.95), sample-hot (T 1.5).
  Default for evaluation: greedy+limits (deterministic) and sample (varied).
- Sampling seed is derived from (seed, prompt): reproducible, and different prompts get independent draws.
- Findings: last.pt (step 600) uses company terms better than best.pt (step 200); T 1.5 breaks text;
  out-of-scope beginnings are turned into company questions.
- Task 6 item A3 corrected to "5+ same tokens in a row" (4 in a row occurs in real questions).
```

Then commit (VS Code terminal, `llm_home`):

```powershell
git status --short
git add src scripts tests prompts docs run_log.md
git status --short
git commit -m "Day 10 home: next-token candidates, sampling, repetition limits, generation report"
git push
```

## Talking points for the review

- **How probabilities become text:** the model never writes a sentence directly. At each step it gives a probability to every token, one token is chosen, it is added to the input, and the model is asked again. The sentence is that loop repeated until `<eos>`.
- **Greedy vs sampling:** greedy always takes the top token, so it is predictable but repetitive (the same beginning always gives the same question). Sampling draws from the probabilities, so it gives variety. With a fixed seed it is still exactly reproducible.
- **Temperature:** the scores are divided by the temperature before softmax. Below 1 the top tokens get even more likely; above 1 the probabilities flatten. The report shows that 1.5 already breaks sentences.
- **top-k and top-p:** both cut away the unlikely tail, where broken pieces and unrelated words live. top-k keeps a fixed number; top-p keeps however many tokens together reach 95%.
- **Why the repetition limits are safe:** they come from measuring the training questions, so the model is only stopped from doing something no real question does.
- **What the candidates prove for task 10:** the mechanism works end to end. For 「今日は」 the model predicts 「誰」 with 72%, which shows it has learned the shape of company questions and nothing else.

## End of Day 10 checklist

- [ ] Files copied (`src\generation.py`, `scripts\day10_generate.py`, `prompts\day10_prompts.txt`, `tests\test_generation.py`, both walkthroughs)
- [ ] pytest: `66 passed`
- [ ] Candidates for 今日は and 営業部の printed and saved to `docs\DAY10_NEXT_TOKEN_CANDIDATES.md`
- [ ] Report written; greedy rows match Day 9 `samples.md`
- [ ] D10 added; committed and pushed
- [ ] **School task 10 complete**

## Send back

1. The pytest result line.
2. The output of Step 3.
3. The "automatic checks" table and the "repetition limits on the untrained model" lines from Step 4.

## Report snippet (Japanese, for `docs/weekly_reports.md`)

```
【課題10】次に続く単語を予測する仕組み
状況: 予定通り終わった(完了)
内容: 学習したモデルが、書き出し(例:「今日は」)に対して次の単語の候補と確率を上位10個まで出す仕組みと、
      その確率から文を作る仕組み(貪欲法と確率による選び方、温度・top-k・top-p、繰り返しの制御)を作成した。
      繰り返しの上限は学習用の質問文から測って決めた。自動テスト12項目を追加し、合計66項目すべて合格。
証拠: docs/DAY10_GENERATION_REPORT.md(候補の表、生成の比較)、src/generation.py、tests/test_generation.py
次: 試す大きさのモデルを学習して比べ、本番のモデルの大きさを決める(作業11)
```
