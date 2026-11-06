# Day 4 (home laptop): generate the question corpus automatically

**Machine:** home laptop. **Repository:** `JC21-240447/llm_home`. **Folder:** `C:\Users\Merrick\Desktop\llm_home`.
**School task:** 5 (text data: making the training text).

## What Day 4 is now, and why

The original plan was "about 40 patterns, about 200,000 lines of company sentences". The team built something better suited to the goal:

| Plan | What the team built |
| --- | --- |
| about 40 sentence patterns | **301 patterns** in **60 question types** (about 5 phrasings per type) |
| statements, dialogues and questions | **questions only**, the way staff ask about employee data |
| about 200,000 lines | **169,727 unique questions**, at most 2,000 per pattern |

Why fewer lines but more patterns: variety matters more than volume. 200,000 lines from 40 patterns would repeat the same 40 sentence shapes thousands of times. 301 patterns teach the model many more ways of asking. The 2,000-per-pattern limit keeps one pattern from flooding the corpus (one type alone could make 1.69 million questions).

Day 4 today means: **run the team's generator yourself, prove that it reproduces the team's corpus, and understand every part of it** so you can explain it to the professor.

### What I already checked

Your `DB.xlsx` (the 50-row table) was not attached, so I built a stand-in table from the names and IDs found in the corpus and ran the generator with it:
- `patterns.csv`, `question_types.csv`, `columns.csv`: **byte-identical** to the team's files.
- 169,727 questions, the same count as the team's corpus.
- 289 of 300 patterns produce **exactly** the team's questions. The 11 that differ all contain people's names. My stand-in lists the names in a different order, so the random choices land on different names.

With your real `DB.xlsx` the whole corpus should come out byte-identical. That is the main check of today.

### SCHOOL-PC SWAP NOTE

CPU only, so the code is the same. On the school PC: install the two new libraries in that PC's `.venv` (Step 2, same commands), and run with `--machine school`.
Do not push `DB.xlsx`, `questions_corpus.zip`, `samples.csv` or the generated files to the **public** school repository before asking me. They contain the 50 names and the internal labels.

---

## Step 1. Put the team's files in place

1. Download `day4_home.zip` from this chat and unzip it.
2. Copy **everything inside** the unzipped `day4_home` folder into `llm_home`. Choose **Replace the files in the destination / ファイルを置き換える** when asked. Replaced on purpose:
   - `requirements.txt`: adds `pandas` and `openpyxl`, which the generator needs to read `DB.xlsx`.
   - `.gitignore` and `LICENSES_AND_SOURCES.md`: the versions for the questions-only corpus.
3. Copy **your `DB.xlsx`** (sheet `employees_v0_50rows_seed42`) into `llm_home\data\source\`.
4. Check (VS Code terminal, `llm_home`):
   ```powershell
   dir data\source
   ```
   Expected: `make_questions.py`, `src_1.txt`, `src_2.txt`, `用語集.csv`, `同義語辞書.csv`, `DB.xlsx`, and the team's outputs `patterns.csv`, `question_types.csv`, `columns.csv`, `slots.csv`, `samples.csv`, `questions_corpus.zip`.
   If `用語集.csv` or `同義語辞書.csv` shows broken characters, rename it by hand. The generator expects exactly these names.

## Step 2. Install pandas and openpyxl (VS Code terminal, `llm_home`)

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -c "import pandas, openpyxl; print(pandas.__version__, openpyxl.__version__)"
.venv\Scripts\python.exe -m pip freeze | Out-File -Encoding ascii requirements_lock_home.txt
```

- The first line installs the two libraries (the others are already there).
- The second line should print two version numbers. pandas 3.0 supports Python 3.11 and newer, so your 3.11.6 is fine.
- The third line refreshes the lock file so it records the exact versions.

## Step 3. Run the generator through the Day 4 script (VS Code terminal, `llm_home`)

```powershell
.venv\Scripts\python.exe scripts\day4_generate_corpus.py --machine home
```

It takes a few seconds. What happens:
1. The inputs are copied into `runs\day4_generation\<date_time>\`. The generator writes its output files into the folder it runs in, so running it directly in `data\source` would overwrite the team's files.
2. The team's generator runs there: `python make_questions.py --corpus 2000 --seed 42`, the same settings the team used.
3. The result is checked and compared with the team's corpus.

Expected output (the important lines):

```text
questions.csv: 169727 questions
patterns: 301 | types: 60 | all distinct questions: 2941425
Questions: 169,727 | types: 60 | patterns with questions: 300 | avg chars: 23.51
Corpus check: PASS (no duplicates, no broken sentences, labels readable)
Same as the team's corpus: byte-identical=True, identical patterns=300, differing patterns=none
patterns.csv: identical
question_types.csv: identical
columns.csv: identical
```

**If `byte-identical=False`:** send me the whole output. It means your `DB.xlsx` differs from the one the team used (for example a different sheet first, or edited rows), or the team regenerated the corpus with changed files. Nothing is broken; we find out which input differs.

Created:

| File | What | In Git? |
| --- | --- | --- |
| `runs/day4_generation/<time>/questions.csv` and the other outputs | the regenerated corpus | no (the team's `questions_corpus.zip` stays the official copy) |
| `runs/day4_generation/<time>/check_result.json` | check results and the SHA-256 of the regenerated corpus | no |
| `docs/DAY4_GENERATION_REPORT.md` | report: inputs with hashes, totals, reproducibility, one example per type | **yes** |
| `run_log.md` | a `Day 4 question generation` entry | **yes** |

## Step 4. Understand the generator (for the review and for the professor's document)

Open `data\source\make_questions.py` in VS Code beside this section. The line numbers below are for that file.

### 4.1 The inputs: everything comes from the team's files

| Input | What the generator takes from it |
| --- | --- |
| `src_1.txt`, `src_2.txt` | the sentence patterns |
| `用語集.csv` | the glossary (read at line 92) |
| `同義語辞書.csv` | variant words for each standard term (lines 93-95), for example 大阪 → 大阪支社 |
| `DB.xlsx` | department and work-location values, the 50 names and employee numbers (lines 96 and 107-114) |

No outside text is used. Every word in a generated question comes from these files or from the small word lists written in the code (for example 高い/低い).

### 4.2 Pattern format (`src_*.txt`)

```text
#T01|条件で一覧
{属性}の社員を一覧で見せてください。||
{属性}の社員を全員教えてください。||
```

- `#T01|条件で一覧` starts a question type: an ID and a name.
- Each following line is one pattern: `sentence|column group|second column group`. The groups are empty when the pattern has no number column.
- `load_patterns()` (lines 232-245) reads the files in order and numbers the patterns P001, P002, ... That gives 301 patterns.
- Words in `{ }` are **slots** to fill. `slots.csv` lists every slot and its rule (written for the team's review).

### 4.3 Slot words and naturalness rules

- **Number columns** (lines 47-71, class `Col`): each column has its own wordings (年収 / 年俸 / 年間給与 / 年間収入), comparison values within the real data range (400万円 to 800万円), adjectives (年収 is 高い/低い, 勤続年数 is 長い/短い), a question word (いくら, 何歳, 何年) and the aggregates that make sense (合計 for money, not for age). `columns.csv` is this table written out.
- **Column groups** (lines 73-81): a pattern says which kind of column fits, for example 大小 (columns with a high/low adjective) or 金額 (money only). So a pattern never asks 「年齢の合計」.
- **Department and place words** (`value_words`, lines 98-105): the standard value plus its synonyms, but only those that read naturally in 「{X}の社員」. Variants containing の, or ending in 課, チーム, 支店 and so on, are skipped.
- **Other slots** (`SLOT_DOMAINS`, lines 153-177): positions, sex, leave reasons, marital status, years, months, age groups, numbers of people.
- **Pairs** (`PAIR_DOMAINS`, lines 178-185): two different values, such as two departments to compare, or a year range where the end year is after the start year.

### 4.4 Building one question (`render`, lines 271-288)

1. Replace each slot with the chosen word.
2. Apply the fixed naturalness rules:
   - 「性別別」 becomes 「男女別」.
   - 「課長の社員」 becomes 「課長」.
   - 「性別が女ではない」 becomes 「女性ではない」.
3. Reject impossible combinations: the same column twice, or asking a named person for their own name.
4. **Stop the whole program** if any slot is left unfilled or a broken join appears (「のの」, 「社員の社員」, 「平均平均」). A broken corpus can never be written silently.

Each question is saved with a **label**: its pattern, question type and the values that filled the slots, as a small JSON text. The labels make it possible to check every question and to review samples (`samples.csv`). They are not part of the training text.

### 4.5 Counting and balancing (`count` and `generate`, lines 290-320)

- `count()` multiplies the number of choices for each slot. All patterns together could make **2,941,425** different questions. They are very uneven: one type could make 1.69 million, another only 5.
- `generate()` takes at most N questions per pattern (N = 2000 with `--corpus 2000`):
  - if the pattern can make at most 3 × N questions, it builds all of them, shuffles them, and keeps N;
  - otherwise it draws random combinations until it has N different questions.
- All randomness uses one random generator with **seed 42**, so the same inputs always give the same corpus. That is what Step 3 proves.

### 4.6 Duplicates across patterns (lines 354-365)

Two patterns can produce the same sentence. The generator keeps only the first copy. This is why pattern **P171** ends up with no questions: every sentence it can make, such as 「部門ごとの月額給与の最高額を教えてください。」, is also made by the earlier pattern **P026**. No sentence is lost, but the team should decide which question type owns these sentences.

### 4.7 Outputs

| File | Content |
| --- | --- |
| `patterns.csv` | 301 patterns: ID, type, style (丁寧 / くだけた / 短い), sentence, column groups, number of possible questions |
| `question_types.csv` | 60 types with pattern counts and totals |
| `columns.csv` | the per-column rules from 4.3 |
| `samples.csv` | 5 example questions per pattern, for the team to check |
| `questions.csv` | the corpus: question, pattern ID, type ID, label |

## Step 5. Look at the result

1. Open `docs\DAY4_GENERATION_REPORT.md` (right-click > **Open Preview / プレビューを開く**). The last table shows one example question per type, which is good material for the professor's document.
2. Open `runs\day4_generation\<time>\samples.csv` in Excel and read the examples for a few types. They are the same sentences the team reviewed.

## Step 6. Commit and push (VS Code terminal, `llm_home`)

```powershell
git status --short
```

`runs/day4_generation/` must **not** appear (it is ignored by `runs/*/`).

```powershell
git add .gitignore LICENSES_AND_SOURCES.md requirements.txt requirements_lock_home.txt run_log.md
git add scripts/day4_generate_corpus.py data/source docs/DAY4_GENERATION_REPORT.md docs/walkthrough
git status --short
git commit -m "Day 4 home: team question generator, reproduction check and report"
git push
```

`DB.xlsx` is included. That is fine here because `llm_home` is private.

## License and attribution (data day)

- `LICENSES_AND_SOURCES.md` now lists only the team's own data: D1 glossary, D2 synonym dictionary, D3 patterns, D4 sample table, D5 generator and corpus. Books and laws are removed.
- Credit line: "Glossary, synonym dictionary, question patterns, sample table and generator: the project team."
- README: on Day 5 you add the line pointing to `docs/DATA_CARD.md`.
- Ask the team (question 5 in my message): are the 50 names in `DB.xlsx` invented? They appear in the training text, so the data card must say so.

## Talking points for the review

- **Why generate the questions instead of writing them by hand:** the model needs many thousands of examples. 16 people cannot write that many by hand, and hand-written text would be uneven. Patterns written by people plus automatic filling give volume and control.
- **Why the words come only from our own files:** there is no copyright question, and every word the model learns is a word our glossary defines.
- **Why 60 types × about 5 phrasings:** each type is one kind of question. Several phrasings per type teach the model that different wordings can mean the same thing. On Day 5 we use that to test on phrasings it never saw.
- **Why a limit per pattern:** without it, one type with 1.69 million possible questions would make up most of the corpus.
- **Why the naturalness rules are in code:** a rule applied by the program is applied to every one of the 169,727 questions. A reviewer could only check a sample.
- **Reproducibility:** fixed seed, fixed inputs, and a byte-for-byte comparison with the team's file, recorded in the run log.
- **Honest limits:** the phrasings are only as varied as the team's patterns, and the people come from a 50-row table, so names repeat.

## End of Day 4 checklist

- [ ] Team files and `DB.xlsx` in `data/source/`
- [ ] `pandas` and `openpyxl` installed; lock file refreshed
- [ ] Script output: `Corpus check: PASS`, 169,727 questions
- [ ] `byte-identical=True` (or the output sent to me if not)
- [ ] Step 4 read with the code open; you can explain 4.2 to 4.6 in your own words
- [ ] Committed and pushed; `runs/day4_generation/` not in Git

## Send back

1. The full console output of the script (as text).
2. If you want the professor's document now: tell me, and I will write it in Japanese (Word and Markdown, the same style as the environment document), based on your real Step 3 result.

## Report snippet (Japanese, for `docs/weekly_reports.md`)

```
【課題5】学習用の文章の作成(質問文の自動生成)
状況: 予定通り終わった
内容: チームが作った用語集・同義語辞書・文型と社員データの表から、質問文を自動で作る
      プログラムを実行し、大量の質問文を作成した。同じ設定で実行すると毎回同じ結果に
      なることを確認した。
証拠: docs/DAY4_GENERATION_REPORT.md、run_log.md、data/source/make_questions.py
次: 作った質問文を、学習用・検証用・テスト用に分ける(Day 5)
```
