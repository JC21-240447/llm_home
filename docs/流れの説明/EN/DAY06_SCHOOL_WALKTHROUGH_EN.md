# Day 6 (school PC): train our own tokenizer (byte-level BPE)

**Machine:** school PC (login `240447`). **Repository:** `JC21-240447/2026_llm_jk3a` (**public**). **Folder:** `C:\Users\240447\Desktop\LLM\phase1_day1`.
**School task:** 8.

The code is **the same** as at home. For what the tokenizer does, why the vocabulary is 1,674 and not 16,000, and the talking points, read `DAY06_HOME_WALKTHROUGH`. What is different at school:
- The files come by SSD.
- **Day 5 must be done at school first**, because the tokenizer trains on `data\corpus\train.txt`, which Day 5 creates on the school PC.
- `git pull` before `git push`.

Only the CPU is used, and training takes seconds.

---

## Step 1. Copy the files from the SSD

At home, put on the SSD (folder `to_school`): `src\bpe_tokenizer.py`, `src\__init__.py`, `scripts\day6_train_bpe.py`, `tests\test_bpe_tokenizer.py` (and `docs\流れの説明\` if you want the walkthroughs).

At school (VS Code terminal, `phase1_day1`; use the SSD's real drive letter):

```powershell
$SSD = "E:\to_school"
New-Item -ItemType Directory -Force src, tests | Out-Null
Copy-Item "$SSD\src\bpe_tokenizer.py", "$SSD\src\__init__.py" src\
Copy-Item "$SSD\scripts\day6_train_bpe.py" scripts\
Copy-Item "$SSD\tests\test_bpe_tokenizer.py" tests\
dir data\corpus\train.txt, data\corpus\val.txt
```

The last command must list both files. If not, do Day 5 at school first.

## Step 2. Tests, training, tests again (VS Code terminal, `phase1_day1`)

```powershell
.venv\Scripts\python.exe -m pytest tests -q
.venv\Scripts\python.exe scripts\day6_train_bpe.py --machine school
.venv\Scripts\python.exe -m pytest tests -q
```

Expected:
- first pytest: `29 passed, 1 skipped`;
- training: 743 pieces, 1,314 merges, chosen 1,674, and `tokenizer.json SHA-256: 53b03761cd1b614972e1b9bfd63d0a66287d6fd069cd0a9fd91419ec249ca94b`, **the same as at home**;
- second pytest: `30 passed`.

The same SHA-256 proves the school PC built exactly the same tokenizer. Take a screenshot. If it differs, do not commit; send me the output.

## Step 3. Record the decision

Add the same **D6** text as in the home walkthrough (Step 6) to `docs\DECISIONS.md`.

## Step 4. Commit and push (VS Code terminal, `phase1_day1`)

```powershell
git status --short
```

Check: nothing under `data/corpus/`, `data/source/` or `runs/` may appear.

```powershell
git add src scripts/day6_train_bpe.py tests data/tokenizer docs/DAY6_TOKENIZER_REPORT.md docs/DECISIONS.md run_log.md
git status --short
git commit -m "Day 6 school: own byte-level BPE tokenizer"
git pull
git push
```

What goes public: the tokenizer code, tests, the report, and the tokenizer files. The tokenizer files hold the learned pieces of our questions (word pieces and the team's invented names), not the questions or the internal labels.

## End of Day 6 (school) checklist

- [ ] Day 5 done at school; files copied from the SSD
- [ ] pytest `29 passed, 1 skipped`, then `30 passed`
- [ ] SHA-256 `53b03761cd1b…`, the same as at home
- [ ] D6 added; committed and pushed; nothing from `data/corpus/`, `data/source/` or `runs/` in Git

## Send back

The console output of the training (as text) and the two pytest result lines.

## Report snippet (Japanese, for `docs/weekly_reports.md`)

```
【課題8】文章を細かく分けるプログラム(トークナイザー)
状況: 予定通り終わった
内容: 学校PCで、自作のバイト単位BPEトークナイザーを学習用の質問文で学習した。語彙の大きさは
      候補ごとに検証用データで測り、決めたルールで選んだ(約1,700)。保存と読み込み、
      文→番号→文の往復が完全に一致することをテストで確認した。
証拠: data/tokenizer/tokenizer.json、docs/DAY6_TOKENIZER_REPORT.md、run_log.md、tests/
次: 課題8・9のデモ(文をトークンと番号に分けて戻す)とコーパスの符号化(Day 7)
```
