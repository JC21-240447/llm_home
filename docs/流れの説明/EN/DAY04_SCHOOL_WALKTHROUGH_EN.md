# Day 4 (school PC): regenerate the questions and prove the result is identical

**Machine:** school PC (login `240447`). **Repository:** `JC21-240447/2026_llm_jk3a` (**public**). **Folder:** `C:\Users\240447\Desktop\LLM\phase1_day1`.
**School task:** 5.

## What Day 4 at school does, and why

On Day 3 you generated the 169,727 questions at home. Today the school PC runs the **same** program on the **same** input files and compares its result with the Day 3 zip. If they are byte-identical, you have proof, on the school PC and in front of anyone, that the questions come from this program and these files.

Only the CPU is used, and the run takes seconds, so this is fine on the school PC.

**Rules you promised the teachers, which still apply:** install only into `.venv`, never with `py -m pip install`, never as administrator, and never install a new Python.

---

## Step 1. Copy the files from the USB stick

Put them into `C:\Users\240447\Desktop\LLM\phase1_day1`:

| From the USB stick | To |
| --- | --- |
| the whole `source` folder (team files and `DB.xlsx`) | `data\source\` |
| `questions_corpus_home_<time>.zip` | `data\source\` |
| `day3_generate_questions.py`, `day4_generate_corpus.py` | `scripts\` |
| `requirements.txt` | the repository root (replace the old one) |

## Step 2. Keep the team's data out of the public repository (do this BEFORE any `git add`)

Open `.gitignore` in VS Code and add these lines at the end:

```text
# Team data files: shared by USB only, never pushed to this public repository
data/source/
```

`runs/*/` is already in `.gitignore`, so the generated files are ignored too. Check (VS Code terminal, `phase1_day1`):

```powershell
git check-ignore -v data\source\DB.xlsx
```

It must print a line that mentions `.gitignore` and `data/source/`. If it prints nothing, the line was not saved: fix it before continuing.

## Step 3. Install pandas and openpyxl into the school `.venv` (VS Code terminal, `phase1_day1`)

```powershell
.venv\Scripts\python.exe -m pip install pandas openpyxl
.venv\Scripts\python.exe -m pip check
.venv\Scripts\python.exe -m pip freeze | Out-File -Encoding ascii requirements_lock_school.txt
```

- This installs into the `.venv` only, exactly as you told 吉澤先生 and 阿保先生.
- `pip check` must print `No broken requirements found.`
- The last line refreshes the lock file, so it records the new versions (pandas supports Python 3.11 and newer, so the school's 3.12 is fine).

If the school network is slow, the download is small (pandas is the largest, tens of MB, not GB).

## Step 4. Run Day 4 against the home zip (VS Code terminal, `phase1_day1`)

Replace `<time>` with the real name of your zip:

```powershell
.venv\Scripts\python.exe scripts\day4_generate_corpus.py --machine school --reference data\source\questions_corpus_home_<time>.zip
```

Expected (the important lines):

```text
Questions: 169,727 | types: 60 | patterns with questions: 300 | avg chars: 23.51
Corpus check: PASS (no duplicates, no broken sentences, labels readable)
Same as the reference corpus (questions_corpus_home_<time>.zip): byte-identical=True, identical patterns=300, differing patterns=none
patterns.csv: identical
question_types.csv: identical
columns.csv: identical
```

If it says `byte-identical=False`, do not change anything. Send me the full output: one of the input files on the USB stick differs from the one used at home.

Take a screenshot of the output. This is the evidence for the professor.

**Optional:** to show the professor the generation itself on the school PC, you can also run Day 3 there, still in seconds:

```powershell
.venv\Scripts\python.exe scripts\day3_generate_questions.py --machine school
```

## Step 5. Commit and push at school (VS Code terminal, `phase1_day1`)

```powershell
git status --short
```

Check carefully before adding: **nothing under `data/source/` and nothing under `runs/`** may appear. If any does, stop and send me the list.

```powershell
git add .gitignore requirements.txt requirements_lock_school.txt run_log.md
git add scripts/day3_generate_questions.py scripts/day4_generate_corpus.py docs/DAY4_GENERATION_REPORT.md
git status --short
git commit -m "Day 4 school: regenerated the question corpus, identical to the home run"
git pull
git push
```

`git pull` before `git push` is needed here, because teammates push to this repository too.

The report `docs/DAY4_GENERATION_REPORT.md` contains example questions with the team's invented names, the input hashes and the totals, but not the internal labels, so it is fine in the public repository.

## Talking points for the review

- **Same program, same inputs, same output, on two machines:** the SHA-256 comparison proves the corpus is reproducible, which is the basis for honest reporting.
- **Why the data stays off the public repository:** the team's table and the internal labels are project data, not something to publish. Only the code and the report are public.
- **Why `pip check` and the lock file again:** the school environment changed (two new libraries), so the evidence the teachers accepted is renewed.

## End of Day 4 (school) checklist

- [ ] Files copied from the USB stick
- [ ] `data/source/` added to `.gitignore`, and `git check-ignore` confirms it
- [ ] pandas and openpyxl installed into `.venv` only; `pip check` OK; lock file refreshed
- [ ] Day 4 output: `byte-identical=True`; screenshot taken
- [ ] Committed and pushed; nothing from `data/source/` or `runs/` in Git

## Send back

1. The full console output of Step 4 (as text) or the screenshot.
2. The `git status --short` list from Step 5, if anything looked unexpected.

## Report snippet (Japanese, for `docs/weekly_reports.md`)

```
【課題5】生成した質問文の確認(学校PCでの再現)
状況: 予定通り終わった
内容: 自宅で生成した質問文と同じプログラム・同じ入力を学校PCで実行し、生成結果が
      1バイトも違わず一致することを確認した。追加のライブラリは仮想環境の中だけに入れた。
証拠: docs/DAY4_GENERATION_REPORT.md、run_log.md、実行画面のスクリーンショット
次: 質問文を学習用・検証用・テスト用に分ける(Day 5)
```
