# Day 3 (home laptop): generate the question corpus, and record it on video

**Machine:** home laptop. **Repository:** `JC21-240447/llm_home`. **Folder:** `C:\Users\Merrick\Desktop\llm_home`.
**School task:** 5 (making the training text).

## What Day 3 is now, and why

The original Day 3 (collecting books and laws) was dropped, because the corpus is now questions only. Day 3 is now **the generation itself**: you run the team's generator `make_questions.py` and get the 169,727 questions. You record it on video as proof that you generated them yourself.

| Day | Where | Job | Proof |
| --- | --- | --- | --- |
| **Day 3** | home (or school) | generate the questions | video, run-log entry, SHA-256 of `questions.csv`, the zip |
| **Day 4** | school | regenerate on the school PC and compare with the Day 3 zip | `byte-identical=True`, run-log entry, report |

Together they show that the questions come from this program and these input files, and that two different machines produce exactly the same result.

**About the reason for running at home:** generation takes only a few seconds (2.7 s on your laptop), so speed is not the reason. If asked, give the true reason: you did the data work at home with the team's files, and the school PC reproduces the same result on Day 4. The run log records the real time, so the explanation must match it.

### SCHOOL-PC SWAP NOTE

Same code. To run Day 3 at school instead: `--machine school`, and pandas/openpyxl must be in the school `.venv` (see the Day 4 school walkthrough, Steps 1 to 3). Do not commit `data/source/` or `runs/` to the public school repository.

---

## Step 1. Put the new files in place

1. Download `day3_home.zip` from this chat and unzip it.
2. Copy **everything inside** the unzipped `day3_home` folder into `llm_home`, and choose **Replace / 置き換える** when asked. It contains:
   - `scripts/day3_generate_questions.py` (new),
   - `scripts/day4_generate_corpus.py` (updated: it can now compare with a Day 3 zip via `--reference`, and it says "reference corpus" instead of "team's corpus"),
   - the Day 3 and Day 4 (school) walkthroughs in `docs/流れの説明/EN` and `JA`.
3. Your `data\source\` already has everything the generator needs, including `DB.xlsx`, from Day 4. pandas and openpyxl are already in your home `.venv`.

## Step 2. Rehearse once without recording (VS Code terminal, `llm_home`)

```powershell
.venv\Scripts\python.exe scripts\day3_generate_questions.py --machine home --no-log
```

`--no-log` keeps the rehearsal out of `run_log.md`. Check that it ends with:

```text
Questions: 169,727 | types: 60 | patterns with questions: 300 | avg chars: 23.51
Check: PASS (no duplicates, no unfilled slots, 60 question types)
questions.csv SHA-256: d9ff2a6a6708f4af81ba3cf8c7db17d88d4112c4e823906f4853ebd1d6d62586
```

That SHA-256 is the hash of the team's `questions.csv`. Your Day 4 run already showed that your laptop produces it byte for byte, so it should match exactly. If it does not, stop and send me the output.

The rehearsal leaves a folder in `runs\day3_generation\`. You can delete it, or leave it, because Git ignores it.

## Step 3. Record the real run

1. **Prepare the screen:** close the browser, chat apps and anything private. Turn on **Do not disturb / 応答不可** (Windows notification settings) so no pop-ups appear.
2. **Start recording** with the **Snipping Tool**: open it, choose **Record / 録画**, select the VS Code window, and press **Start / スタート**. Xbox Game Bar (**Win + Alt + R**) also works and records the active window.
3. **In the VS Code terminal, run these one after another**, slowly enough that each output can be read:
   ```powershell
   Get-Date
   hostname
   git log -1 --oneline
   dir data\source
   .venv\Scripts\python.exe scripts\day3_generate_questions.py --machine home
   ```
   - `Get-Date` and `hostname` show when the run happened and on which PC.
   - `git log -1 --oneline` shows which version of the code is in the folder.
   - `dir data\source` shows the input files: the team's glossary, synonym dictionary, patterns and table.
   - The last line is the real run. Wait until `Appended an entry to run_log.md` appears.
4. **Show the result:**
   - Scroll up slowly over the **10 example questions** and the `Check: PASS` line.
   - In the Explorer, open `runs\day3_generation\<time>\questions.csv` (VS Code shows it as text) and scroll through a few hundred lines.
   - Open `run_log.md` and show the new `Day 3 question generation` entry at the bottom.
5. **Stop recording.** The video is saved in `C:\Users\Merrick\Videos\Captures` (or `ビデオ\キャプチャ`).
6. **Keep the video outside the repository** (for example on your SSD or in cloud storage). Videos are large and do not belong in Git.

About 3 to 5 minutes is enough. Do not speed it up or cut the middle out, because an uncut recording is better proof.

## Step 4. Prepare what goes to school

In `runs\day3_generation\<time>\` you now have:

| File | Use |
| --- | --- |
| `questions_corpus_home_<time>.zip` | **take this to school by USB**; Day 4 compares the school result with it |
| `questions.csv` | the generated questions |
| `generation_info.json` | settings, the SHA-256 of every input and of `questions.csv`, the time |

Copy to the USB stick:
- `runs\day3_generation\<time>\questions_corpus_home_<time>.zip`,
- the whole `data\source\` folder (the team's files and `DB.xlsx`),
- `scripts\day3_generate_questions.py` and `scripts\day4_generate_corpus.py`,
- `requirements.txt` (it lists pandas and openpyxl).

**Do not upload these through the public school repository.** `data\source\` and the zip contain the team's table and the internal labels. USB only.

## Step 5. Commit and push at home (VS Code terminal, `llm_home`)

```powershell
git status --short
git add scripts run_log.md docs
git status --short
git commit -m "Day 3 home: question generation script, run and walkthroughs"
git push
```

`runs/` must not appear in the list (the generated files are ignored).

## Talking points for the review

- **What Day 3 proves:** the 169,727 questions are produced by the team's program from the team's files, and you ran it yourself. The video shows the run, and the run log and `generation_info.json` record the inputs, settings and output hash.
- **Why the output hash matters:** a SHA-256 hash changes completely if even one character changes. When the school PC produces the same hash on Day 4, the two corpora are identical, not merely similar.
- **Why the run is fast:** the generator only fills patterns with words and removes duplicates. That is light work for a CPU and needs no GPU.
- **Why the inputs are copied to a separate folder:** the generator writes its outputs where it runs. A separate folder keeps the team's files untouched and keeps each run's results together.
- **Why the video is uncut:** an uncut recording with the date, the PC name and the code version visible is hard to question.

## End of Day 3 checklist

- [ ] Rehearsal passed and showed SHA-256 `d9ff2a6a6708…`
- [ ] Real run recorded uncut: `Get-Date`, `hostname`, `git log`, `dir`, the run, the questions, the run-log entry
- [ ] Video saved outside the repository
- [ ] USB stick has the zip, `data\source\`, the two scripts and `requirements.txt`
- [ ] Committed and pushed; `runs/` not in Git

## Send back

1. The console output of the real run (as text).
2. Tell me when the Day 4 school run is done, and send its output too.

## Report snippet (Japanese, for `docs/weekly_reports.md`)

```
【課題5】学習用の文章の作成(質問文の生成)
状況: 予定通り終わった
内容: チームの用語集・同義語辞書・文型・社員データの表をもとに、質問文を自動で作る
      プログラムを自宅のPCで実行し、大量の質問文を生成した。実行の様子は動画で記録し、
      入力と出力のハッシュ値を実行ログに残した。
証拠: 実行の動画、run_log.md、runs/day3_generation/ の generation_info.json
次: 学校PCで同じプログラムを実行し、同じ結果になることを確認する(Day 4)
```
