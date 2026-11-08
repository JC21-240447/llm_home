# Day 5 (home laptop): split the question corpus by pattern, and write the data card

**Machine:** home laptop. **Repository:** `JC21-240447/llm_home`. **Folder:** `C:\Users\Merrick\Desktop\llm_home`.
**School task:** 5 (text data and corpus).

## What changed, and why

The team finished the data work in a separate chat, so the plan changed (see the handoff file):

| Walkthrough Day | Now |
| --- | --- |
| Day 2: glossary and synonym files | **Done by the team**: 551 terms, 1,705 variants |
| Day 3: books and laws | **Dropped**: the corpus is questions only |
| Day 4: synthetic text generator | **Done by the team**: 60 question types, 301 patterns, 169,727 questions |
| **Day 5: split and data card** | **Today** |

Today's job is to turn the team's corpus into **train, validation and test files** without cheating ourselves.

The key rule: **split by pattern, never by line.** A pattern is one sentence frame, such as 「{属性}の社員を一覧で見せてください。」. Its questions differ only in the filled-in words, so they are almost the same sentence. If lines were split at random, nearly the same sentence would be in train and test, and the test score would look good for the wrong reason. So we hold out **whole patterns**. The model is tested on phrasings it has never seen, for question types it has seen.

### What the script decides, and why

| Rule | Value | Reason |
| --- | --- | --- |
| Held-out patterns | 21 for validation, 21 for test (about 7% + 7% of 300) | Enough phrasings to measure, while train keeps most of the data |
| One pattern per type, val and test types different | 42 of 60 types get a held-out phrasing | Wide coverage of types |
| A type may donate only if it has 100+ questions and 3+ patterns | | Small types keep all their phrasings for training |
| The held-out pattern may hold at most 40% of its type | | Without this rule, T50 would have kept 16 lines in train and sent 256 to test (I caught this while testing) |
| Cap per held-out pattern | 500 questions | One big pattern cannot dominate the score. The lines over the cap are dropped, never moved to train, because that would leak. |
| Near-duplicate removal | same text ignoring punctuation and spaces | Guards against 「…ですか。」 and 「…ですか？」 pairs crossing splits (result: 0 found) |
| Oversampling | small types repeated up to about 200 lines, at most 5 copies, **train only** | "A few times, not hundreds", as the handoff says |
| Seed | 42 | The same input always gives byte-identical output |

### Two findings from checking the team's files

1. **P171 has no questions.** Every sentence it makes, for example 「部門ごとの月額給与の最高額を教えてください。」, is also made by **P026** (T06, 集計 = 最高額). The generator keeps the first copy, so all P171 questions are filed under P026. Training loses nothing, but the same sentence has two possible type labels. The team should decide which type owns it (a question for later, it does not block today).
2. **Every one of the 169,727 questions matches its pattern's template**, allowing for the rule that turns 「課長の社員」 into 「課長」. The corpus and `patterns.csv` agree.

### SCHOOL PC

Day 5 uses only the CPU and plain Python, so the code is identical on the school PC. The school steps are different, because the school repository is public: follow `DAY05_SCHOOL_WALKTHROUGH`.

---

## Step 1. Put the files in place

Everything Day 5 reads is already in `data\source\` from Days 3 and 4, and `.gitignore` and `LICENSES_AND_SOURCES.md` were already updated on Day 4. Only the script and the walkthroughs are new.

1. Download `day5_v2.zip` from this chat and unzip it.
2. Copy **everything inside** the unzipped `day5_v2` folder into `llm_home`:
   - `scripts\day5_build_corpus.py`,
   - `docs\流れの説明\EN\` and `JA\`: the Day 5 home and school walkthroughs.
3. Check (VS Code terminal, `llm_home`):
   ```powershell
   dir data\source\questions_corpus.zip, data\source\patterns.csv
   git check-ignore -v data\corpus\train.txt
   ```
   The first command must list both files. The second must print a line mentioning `data/corpus/`, which means the generated corpus will not go into Git.

## Step 2. Run the split (VS Code terminal, folder `llm_home`)

```powershell
.venv\Scripts\python.exe scripts\day5_build_corpus.py --machine home
```

It takes about 10 seconds. You should see exactly these numbers (I ran the same script on the same files):

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

Then open `data\splits\split_manifest.json` and look at `outputs_sha256`. On my test run (Python 3.11 and 3.13, same result) the three text files had these hashes:

| File | SHA-256 begins with |
| --- | --- |
| `data/corpus/train.txt` | `21d5d2900c71` |
| `data/corpus/val.txt` | `bc5a13145d1b` |
| `data/corpus/test.txt` | `c0c99ebdd755` |

If yours match, your files are byte-for-byte the same as mine, which shows the split depends only on the data and the seed. The script also checks itself: it stops with `Split check FAILED` if any held-out line leaked into train.

If the numbers differ from the block above, do not continue. Send me the output.

What was created:

| File | What it is | In Git? |
| --- | --- | --- |
| `data/corpus/train.txt`, `val.txt`, `test.txt` | one question per line, text only: the training input | no (rebuilt by the script) |
| `data/corpus/*_labels.csv` | the same lines with pattern, type and 意味, internal only | no |
| `data/splits/split_manifest.json` | which patterns went where, settings, SHA-256 of inputs and outputs | **yes** |
| `docs/DATA_CARD.md` | the data card, all numbers written by the script | **yes** |
| `run_log.md` | a new `Day 5 corpus split` entry | **yes** |

## Step 3. Read the result

1. Open `docs/DATA_CARD.md` in VS Code (right-click > **Open Preview / プレビューを開く** shows the tables).
2. In section 6 (per question type), check that for every type with a held-out pattern, **train still has more lines than validation plus test**.
3. Open `data\corpus\val.txt` and read about 20 lines. They should be natural questions in phrasings you can find in `data\source\patterns.csv` under the pattern IDs listed in the manifest's `patterns_per_split.val`.
4. Understand the token numbers. We have no tokenizer yet (Day 6), so the script gives honest **bounds**:
   - **Lower bound (about 1.65 million for train):** the planned pre-tokenizer splits text into runs of the same script (hiragana, katakana, kanji, Latin, digits, punctuation). BPE never merges across these runs, so there are at least that many tokens.
   - **Upper bound (about 10.1 million):** UTF-8 bytes, which would be the count if BPE learned nothing.
   - Day 7 measures the real number. It will be closer to the lower bound.

## Step 4. Record the decisions

Add this to the end of `docs/DECISIONS.md`:

```markdown
## D4. Training corpus (Day 5)

- The corpus is short Japanese questions about employee data only, made by the team's own generator
  from the team's glossary, synonym dictionary, patterns and a 50-row sample table.
- Books and laws (old Day 3) are dropped. General text may be reconsidered only if evaluation shows a need.
- Only the question text is used for training; labels are kept in separate files.

## D5. Split (Day 5)

- Split by pattern, not by line: 21 validation and 21 test patterns, one per question type, from different types.
- A type donates a pattern only if it has 100+ questions and 3+ patterns; the donated pattern holds at most 40% of its type.
- Held-out patterns are capped at 500 questions; lines over the cap are dropped, not moved to train.
- Small types are repeated in train only (up to about 200 lines, at most 5 copies). Seed 42.
- P171 has no questions: all its sentences are identical to P026 sentences. Open question for the team.
```

Also add this line to the "Data and licenses" section of `README.md`:

```markdown
All training text is made by the team's own generator from the team's own files; see `docs/DATA_CARD.md`.
```

## Step 5. Commit and push (VS Code terminal, `llm_home`)

```powershell
git status --short
```

Check the list before adding anything: `data/corpus/` must **not** appear (it is ignored), and nothing from `.venv`.

```powershell
git add README.md run_log.md
git add scripts/day5_build_corpus.py data/splits docs
git status --short
git commit -m "Day 5 home: team data, split by pattern, data card"
git push
```

## Talking points for the review

- **Why split by pattern:** lines from one pattern are near-copies. A random split tests memory, not generalisation. Holding out whole phrasings tests whether the model handles wording it never saw.
- **Why one pattern per type, and the 40% rule:** we want evaluation across many types, without starving any type in training. The first version of the script broke this for one type (16 training lines against 256 test lines). The rule exists because of that test, and it is a good example of checking a design on real data.
- **Why the cap drops lines instead of moving them to train:** moving held-out phrasings into train would leak exactly what we want to test.
- **Why oversample, and why only a little:** some types can make only 5 to 70 distinct questions. Repeating them a few times stops the model from ignoring them. Repeating them hundreds of times would make it memorise them.
- **Reproducibility:** the split depends only on the input files and the seed. The manifest stores SHA-256 hashes, so anyone can prove that the school PC and the laptop used identical data.
- **Licensing:** every line was made by the team's own generator from the team's own files. No outside text is used for training.
- **Token bounds:** we report a provable range now, and replace it with the measured count on Day 7. We do not invent a number.

## End of Day 5 checklist

- [ ] Day 5 script copied; `data/corpus/` confirmed as ignored
- [ ] Script output matches the numbers in Step 2; `Split check: PASS`
- [ ] The three `.txt` hashes match the table in Step 2
- [ ] `docs/DATA_CARD.md` read; train is bigger than validation plus test for every held-out type
- [ ] `docs/DECISIONS.md` (D4, D5) and the README line added
- [ ] Committed and pushed; `data/corpus/` not in Git

## Send back

1. The full console output of the script (as text).
2. The `outputs_sha256` block from `data/splits/split_manifest.json`.
3. Answers to my open questions (in the chat message).

## Report snippet (Japanese, for `docs/weekly_reports.md`)

Following the report already submitted, the corpus size is described in general terms, without exact numbers.

```
【課題5】文章データ(コーパス)の準備
状況: 予定通り終わった
内容: チームが作った用語集・同義語辞書・文型から作成した大量の質問文を、文型ごとに
      学習用・検証用・テスト用に分けた。同じ文型の文がテストに混ざらないようにし、
      一度も見ていない言い回しで評価できるようにした。データカードを作成した。
証拠: docs/DATA_CARD.md、data/splits/split_manifest.json、run_log.md
次: 課題8(文章を細かく分けるプログラム = トークナイザー)
```
