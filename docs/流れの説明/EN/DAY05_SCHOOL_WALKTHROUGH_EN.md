# Day 5 (school PC): split the question corpus by pattern, and write the data card

**Machine:** school PC (login `240447`). **Repository:** `JC21-240447/2026_llm_jk3a` (**public**). **Folder:** `C:\Users\240447\Desktop\LLM\phase1_day1`.
**School task:** 5.

The code is **the same** as at home: `scripts\day5_build_corpus.py`, run with `--machine school`. What is different at school:
- The repository is public, so the generated corpus (`data/corpus/`, which also contains the internal labels) must be added to `.gitignore` first.
- `LICENSES_AND_SOURCES.md` at school is still the old version (books and laws), so it is replaced.
- `git pull` before `git push`, because teammates push to this repository.

For the reasons behind the split (why by pattern, the 40% rule, the cap, oversampling), read the home walkthrough `DAY05_HOME_WALKTHROUGH`. Everything there applies here too.

**Do Day 3 and Day 4 at school first.** Day 5 reads `data\source\questions_corpus.zip` and `data\source\patterns.csv`, which you copied there for Day 3 and Day 4.

---

## Step 1. Copy the files from the SSD

At home, put on the SSD (folder `to_school`):
- `scripts\day5_build_corpus.py`
- `LICENSES_AND_SOURCES.md`
- `docs\流れの説明\` (optional; the walkthroughs)

At school (VS Code terminal, `phase1_day1`; use the SSD's real drive letter):

```powershell
$SSD = "E:\to_school"
Copy-Item "$SSD\scripts\day5_build_corpus.py" scripts\
Copy-Item "$SSD\LICENSES_AND_SOURCES.md" .
dir data\source\questions_corpus.zip, data\source\patterns.csv
```

The last command must list both files. If not, do Day 3 and Day 4 at school first.

## Step 2. Keep the generated corpus out of the public repository (before running anything)

Open `.gitignore` and add at the end (under the `data/source/` line you added on Day 4):

```text
# Day 5: generated corpus files with internal labels (rebuild with scripts/day5_build_corpus.py)
data/corpus/
```

Check:

```powershell
git check-ignore -v data\corpus\train.txt
```

It must print a line mentioning `data/corpus/`.

## Step 3. Run the split (VS Code terminal, `phase1_day1`)

```powershell
.venv\Scripts\python.exe scripts\day5_build_corpus.py --machine school
```

Expected, exactly the same numbers as at home:

```text
Loaded 169,727 questions, 300 patterns with questions, 60 types.
Patterns with no questions (...): ['P171']
split      lines    unique       chars              token bounds  patterns  types
train    148,860   147,317   3,482,192    1,646,022 - 10,126,458       258     60
val        4,517     4,517      95,987          42,388 - 279,399        21     21
test       4,553     4,553     109,720          51,015 - 320,290        21     21
Near-duplicates removed: {'val': 0, 'test': 0}; over eval cap: {'val': 5700, 'test': 7640}
Split check: PASS (no shared patterns, no near-duplicate leaks)
```

Then open `data\splits\split_manifest.json` and check `outputs_sha256`:

| File | SHA-256 begins with |
| --- | --- |
| `data/corpus/train.txt` | `21d5d2900c71` |
| `data/corpus/val.txt` | `bc5a13145d1b` |
| `data/corpus/test.txt` | `c0c99ebdd755` |

The same hashes as at home prove that the school PC built exactly the same training, validation and test files. Take a screenshot of the output and of these three lines.

If anything differs, do not commit. Send me the output.

## Step 4. Record the decisions

Add the same **D4** and **D5** text as in the home walkthrough (Step 4) to the end of `docs\DECISIONS.md`, and the same line to the "Data and licenses" section of `README.md`.

## Step 5. Commit and push (VS Code terminal, `phase1_day1`)

```powershell
git status --short
```

Check: **nothing under `data/corpus/`, `data/source/` or `runs/`** may appear. If anything does, stop and send me the list.

```powershell
git add .gitignore LICENSES_AND_SOURCES.md README.md run_log.md
git add scripts/day5_build_corpus.py data/splits docs/DATA_CARD.md docs/DECISIONS.md
git status --short
git commit -m "Day 5 school: split by pattern, data card"
git pull
git push
```

What goes public, and why it is fine: the script, the manifest (pattern IDs, settings, counts and hashes), the data card (counts and design, no labels) and the decisions. The questions, labels and team table stay on the school PC and the SSD.

## End of Day 5 (school) checklist

- [ ] Day 3 and Day 4 already done at school
- [ ] `data/corpus/` added to `.gitignore` and confirmed with `git check-ignore`
- [ ] Output numbers match; `Split check: PASS`; the three hashes match the table
- [ ] D4, D5 and the README line added
- [ ] Committed and pushed; nothing from `data/corpus/`, `data/source/` or `runs/` in Git

## Send back

The full console output (as text) and the `outputs_sha256` lines.

## Report snippet (Japanese, for `docs/weekly_reports.md`)

```
【課題5】文章データ(コーパス)の準備
状況: 予定通り終わった
内容: 学校PCで、生成した大量の質問文を文型ごとに学習用・検証用・テスト用に分けた。
      同じ文型の文がテストに混ざらないようにし、一度も見ていない言い回しで評価できる
      ようにした。データカードを作成した。
証拠: docs/DATA_CARD.md、data/splits/split_manifest.json、run_log.md
次: 課題8(文章を細かく分けるプログラム = トークナイザー)
```
