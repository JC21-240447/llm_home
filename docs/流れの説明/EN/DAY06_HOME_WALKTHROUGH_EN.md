# Day 6 (home laptop): train our own tokenizer (byte-level BPE)

**Machine:** home laptop. **Repository:** `JC21-240447/llm_home`. **Folder:** `C:\Users\Merrick\Desktop\llm_home`.
**School task:** 8 (the splitting program: text in, pieces out). The demo for tasks 8 and 9 follows on Day 7.

## What Day 6 does, and why

A language model cannot read characters. It reads **token IDs**: numbers that stand for pieces of text. The tokenizer decides how text is cut into pieces and which number each piece gets. Today we write it ourselves and train it on the Day 5 training text.

How it works, in three layers:

1. **Pre-tokenizer.** The text is cut wherever the kind of character changes: hiragana, katakana, kanji, Latin letters, spaces. Digits and punctuation are always single pieces.
   `営業部の社員は何人ですか。` becomes `営業部 | の | 社員 | は | 何人 | ですか | 。`
2. **Bytes.** Each piece becomes its UTF-8 bytes (a Japanese character is 3 bytes). All 256 byte values have their own token, so **any** text can be encoded, including words we never saw, English, symbols and emoji. There are no "unknown words".
3. **BPE merges.** Training counts every pair of neighbouring tokens, merges the most frequent pair into a new token, and repeats. Frequent words end up as one token. Rare words stay as several smaller tokens.

Fixed ID layout, decided in the plan and kept unchanged:

| IDs | Meaning |
| --- | --- |
| 0, 1, 2, 3 | `<pad>`, `<bos>` (start), `<eos>` (end), `<unk>` |
| 4 to 103 | `<unused_0>` to `<unused_99>`: a spare block of IDs for later extension |
| 104 to 359 | the 256 single bytes |
| 360 and up | the learned merges, in the order they were learned |

### The big finding: the vocabulary size

The plan said about 16,000 tokens. The data says that is impossible here. The 169,727 questions break into only **743 different pieces** (`営業部`, `の`, `社員`, `えてください`, ...). BPE cannot merge across pieces, so training learns **1,314 merges and then stops by itself**: nothing else repeats. The largest possible vocabulary is therefore **1,674**.

Because merges are learned in order, **the first N merges are exactly the size-N tokenizer**, so one training run lets us measure every size:

| Vocabulary | Validation tokens | Key terms at 1-3 tokens | Longest question |
| --- | --- | --- | --- |
| 400 | 168,640 | 16/232 | 79 |
| 800 | 72,159 | 168/232 | 45 |
| 1,200 | 51,888 | 200/232 | 26 |
| 1,600 | 47,129 | 225/232 | 25 |
| **1,674 (chosen)** | **47,123** | **232/232** | **25** |

**The rule, written down before choosing:** take the smallest size where every achievable key term (column names, values and query words that can be 1 to 3 tokens) is 1 to 3 tokens, and the validation text needs at most 5% more tokens than the largest tokenizer. Only 1,674 satisfies the first part.

Two more numbers to keep:
- **The longest question is 25 tokens** (27 with start and end). That is evidence for the context-length decision later.
- **Training takes about 2 seconds**, so our own pure-Python implementation is fast enough. The plan's fallback (a ready-made tokenizer library) is not needed.

### What 「今日は晴れ」 shows

The school task sheet uses 「今日は晴れ」 as its example. Our real output is:

```text
今日は晴れ -> <e4bb> / <8a> / 日 / は / <e699> / <b4> / <e382> / <8c>
```

今日 and 晴れ **never appear** in our questions, so the tokenizer has no tokens for them. It writes them with byte tokens (shown as `<e4bb>`) and still decodes them back exactly. Show this honestly: it demonstrates both that the tokenizer learned **our** vocabulary, and that it can still handle any text. For domain sentences the split looks like the school's example:

```text
営業部の社員の平均年収を教えてください。 -> 営業部 / の / 社員 / の / 平均年収 / を / 教 / えてください / 。
```

### SCHOOL PC

Same code, CPU only. The school steps are in `DAY06_SCHOOL_WALKTHROUGH`. The school PC must produce the same `tokenizer.json` SHA-256.

---

## Step 1. Put the files in place

1. Download `day6.zip` from this chat and unzip it.
2. Copy **everything inside** the unzipped `day6` folder into `llm_home`:
   - `src\bpe_tokenizer.py`: the tokenizer (reused from now on by every later Day),
   - `src\__init__.py`: an empty file that lets scripts import from `src`,
   - `scripts\day6_train_bpe.py`: training, measuring, choosing, saving, report,
   - `tests\test_bpe_tokenizer.py`: unit tests,
   - `docs\流れの説明\EN` and `JA`: the Day 6 walkthroughs.

## Step 2. Run the unit tests first (VS Code terminal, `llm_home`)

```powershell
.venv\Scripts\python.exe -m pytest tests -q
```

Expected: `29 passed, 1 skipped`. The skipped test needs the real tokenizer, which does not exist yet. The tests check the fixed ID layout, that the pre-tokenizer loses nothing, deterministic training, exact round trips on tricky text (emoji, English, tabs, line breaks, half-width katakana), and that saving and loading give identical results.

## Step 3. Train and choose (VS Code terminal, `llm_home`)

```powershell
.venv\Scripts\python.exe scripts\day6_train_bpe.py --machine home
```

It takes a few seconds. Expected (the important lines):

```text
Unique pre-token pieces: 743 (total 1,730,834)
Merges learned: 1,314 -> largest vocabulary 1,674 (most frequent pair occurs fewer than 2 times)
Key terms found in the training text: 279; achievable as 1-3 tokens: 232
Rule: smallest size with all achievable key terms at 1-3 tokens and validation within 5% of the largest -> chosen 1,674
Reload check: PASS (identical merges and encodings); round trip: PASS on 4,523 texts
tokenizer.json SHA-256: 53b03761cd1b614972e1b9bfd63d0a66287d6fd069cd0a9fd91419ec249ca94b
```

The SHA-256 must be exactly this, because training is deterministic and your Day 5 files are byte-identical to mine. If it differs, send me the output.

Created:

| File | What | In Git? |
| --- | --- | --- |
| `data/tokenizer/tokenizer.json` | the chosen tokenizer (1,674) | **yes** |
| `data/tokenizer/tokenizer_full.json` | all merges (here the same size; kept for re-saving other sizes) | **yes** |
| `docs/DAY6_TOKENIZER_REPORT.md` | the full table, the rule, the demo, hashes | **yes** |
| `run_log.md` | a `Day 6 tokenizer` entry | **yes** |

## Step 4. Run the tests again

```powershell
.venv\Scripts\python.exe -m pytest tests -q
```

Now expected: `30 passed`. The last test encodes and decodes all 4,517 validation questions with the real tokenizer and checks that each one comes back exactly.

## Step 5. Read the report

Open `docs\DAY6_TOKENIZER_REPORT.md` in preview. Check three things:
1. The size table: tokens drop quickly up to about 1,400 and then barely change. That is the evidence for "small vocabulary".
2. "Achievable key terms longer than 3 tokens at the chosen size: none".
3. The 6 key terms that stay longer than 3 tokens and why: they are phrases of several pieces (数が多い順), or words that only appear inside longer words (勤続 only inside 勤続年数).

## Step 6. Record the decision

Add to the end of `docs\DECISIONS.md`:

```markdown
## D6. Tokenizer (Day 6)

- Own byte-level BPE with a script-class pre-tokenizer (digits and punctuation always single pieces). No tokenizer library.
- Fixed IDs: <pad>=0, <bos>=1, <eos>=2, <unk>=3, <unused_0..99>=4-103, bytes=104-359, merges from 360.
- Vocabulary 1,674 instead of the planned ~16,000: the training text has only 743 unique pieces, so training stops
  after 1,314 merges. Chosen by a written rule (all achievable key terms 1-3 tokens, validation within 5% of the
  largest size); see docs/DAY6_TOKENIZER_REPORT.md.
- Evidence for later: the longest question is 25 tokens (27 with <bos>/<eos>). The context length is decided later.
- Trained only on train.txt; validation used only for measuring.
```

## Step 7. Commit and push (VS Code terminal, `llm_home`)

```powershell
git status --short
git add src scripts tests data/tokenizer docs run_log.md
git status --short
git commit -m "Day 6 home: own byte-level BPE tokenizer, size chosen from evidence"
git push
```

`data/corpus/` must not appear in the list.

## Talking points for the review

- **Why bytes:** with 256 byte tokens, every possible text can be encoded. There is no "unknown word" problem, even for English, symbols or emoji.
- **Why a pre-tokenizer:** cutting at changes of character type keeps tokens meaningful (a token never mixes a kanji word with the particle after it) and makes training fast.
- **Why digits are single pieces:** any number can be written digit by digit, so the model never meets a number it has no token for.
- **Why the spare IDs (`<unused_0..99>`):** a reserved block of IDs for later extension, so the embedding table never has to change size.
- **Why 1,674 and not 16,000:** we measured it. The text has only 743 different pieces, so a bigger vocabulary cannot exist. Choosing by a written rule, with the table in the report, is more defensible than choosing a round number.
- **Why train only on train.txt:** if validation text shaped the vocabulary, validation scores would be flattered. We use it only to measure.
- **What 今日は晴れ shows:** the tokenizer knows our domain words. Unknown words are written as bytes and still come back exactly. Show the real output, not the one on the task sheet.
- **Deterministic:** the same training text always gives the same file (same SHA-256), on any PC.

## End of Day 6 checklist

- [ ] Files copied (`src`, `scripts`, `tests`, docs)
- [ ] Tests before training: `29 passed, 1 skipped`
- [ ] Training: 743 pieces, 1,314 merges, chosen 1,674, SHA-256 `53b03761cd1b…`
- [ ] Tests after training: `30 passed`
- [ ] Report read; D6 added to `docs/DECISIONS.md`
- [ ] Committed and pushed

## Send back

1. The console output of Step 3 (as text).
2. The pytest result line of Step 4.

## Report snippet (Japanese, for `docs/weekly_reports.md`)

```
【課題8】文章を細かく分けるプログラム(トークナイザー)
状況: 予定通り終わった
内容: バイト単位のBPEトークナイザーを自作し、学習用の質問文で学習した。語彙の大きさは、
      候補ごとに検証用データで測り、決めたルールで選んだ(約1,700)。保存と読み込み、
      文→番号→文の往復が完全に一致することをテストで確認した。
証拠: data/tokenizer/tokenizer.json、docs/DAY6_TOKENIZER_REPORT.md、run_log.md、tests/
次: 課題8・9のデモ(文をトークンと番号に分けて戻す)とコーパスの符号化(Day 7)
```
