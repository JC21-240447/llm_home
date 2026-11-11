# Day 7 (home laptop): tokenizer tests, the text → numbers → text demo, and the encoded corpus

**Machine:** home laptop. **Repository:** `JC21-240447/llm_home`. **Folder:** `C:\Users\Merrick\Desktop\llm_home`.
**School plan:** finishes **task 8** (its demo) and **task 9** (turning pieces into numbers and back). After today, task 9 is complete, so its document can be written.

## What Day 7 does, and why

Day 6 built the tokenizer. Today we prove it is correct on all our data and prepare the data for training:

1. **A demo** (`day7_tokenizer_demo.py`) that shows, for each sentence, the pieces, the number of every piece, and the text decoded back from the numbers. This is the evidence for school tasks 8 and 9.
2. **A full check** (`day7_encode_corpus.py`): every one of the questions in train, validation and test is encoded and decoded, and must come back identical. So must 9 tricky texts (emoji, English, symbols, line breaks, a 4-byte kanji).
3. **Encoding the corpus** into binary token files. Each split becomes one long stream of numbers, `<bos> question <eos> <bos> question <eos> …`, saved as 16-bit integers in `data\tokens\`. The trainer (Day 9) reads these files directly, piece by piece, without loading everything into memory. After writing, the script reads the files back from disk and decodes them, to prove the files themselves are right.
4. **The real token counts.** Day 5 could only give a range (1.65 to 10.1 million for train). Now we measure: **2,028,601 tokens for train**. This number decides the model size on Day 11.

New code:

| File | What |
| --- | --- |
| `src\token_data.py` | writing and reading token files (Day 9 reuses it) |
| `scripts\day7_encode_corpus.py` | full round-trip check, term table, encoding, read-back, report |
| `scripts\day7_tokenizer_demo.py` | the text → pieces → numbers → text demo |
| `tests\test_token_data.py` | 4 new tests (stream layout, file round trip, size check, real files) |

### SERVER NOTE (later)

All Day 7 code is plain Python + NumPy and runs unchanged on Linux. On the server, use `/` instead of `\` and the Linux venv path, for example `.venv/bin/python scripts/day7_encode_corpus.py --machine server`. The token files are written in a fixed byte order, so the server must produce the same SHA-256 values as your laptop.

---

## Step 1. Put the files in place

1. Download `day7.zip` from this chat and unzip it.
2. Copy **everything inside** the unzipped `day7` folder into `llm_home`. Check the destination folders this time: `src\token_data.py` goes into **`llm_home\src`**, not `data\source`.
3. Check (VS Code terminal, `llm_home`):
   ```powershell
   dir src, scripts\day7_*.py, tests
   ```
   `src` must show `bpe_tokenizer.py`, `token_data.py` and `__init__.py`. `tests` must show both test files.

## Step 2. Tests before encoding (VS Code terminal, `llm_home`)

```powershell
.venv\Scripts\python.exe -m pytest tests -q
```

Expected: `33 passed, 1 skipped`. The skipped test checks the real token files, which do not exist yet.

## Step 3. Encode the corpus (VS Code terminal, `llm_home`)

```powershell
.venv\Scripts\python.exe scripts\day7_encode_corpus.py --machine home
```

It takes about 10 seconds. Expected (the important lines):

```text
Round trip on 9 tricky texts: PASS
train   148,860 questions   2,028,601 tokens  avg 13.6  max  28  partial-char    3 in 1 questions  -> round trip + read-back PASS
val       4,517 questions      56,157 tokens  avg 12.4  max  22  partial-char  503 in 231 questions  -> round trip + read-back PASS
test      4,553 questions      63,877 tokens  avg 14.0  max  25  partial-char    0 in 0 questions  -> round trip + read-back PASS
Key terms: 772 (in training text: 279); 1-3 tokens: 464 of all, 273 of those in the training text
Total tokens: 2,148,635 (train 2,028,601)
```

Open `docs\DAY7_ENCODING_REPORT.md` and check the hashes at the bottom:

| File | SHA-256 begins with |
| --- | --- |
| `data/tokens/train.bin` | `2c95990940ea` |
| `data/tokens/val.bin` | `e39257300d28` |
| `data/tokens/test.bin` | `ec593af0f94a` |

The same hashes as mine prove the files are byte-identical. If anything differs, send me the output.

## Step 4. Tests after encoding

```powershell
.venv\Scripts\python.exe -m pytest tests -q
```

Expected: `34 passed`. The last test reads the real validation token file and checks that every question decodes back exactly.

## Step 5. The demo for tasks 8 and 9

```powershell
.venv\Scripts\python.exe scripts\day7_tokenizer_demo.py
```

For each sentence it prints the pieces, the numbers and the decoded text. For example:

```text
Input   : 営業部の社員の平均年収を教えてください。
Pieces  : 営業部 / の / 社員 / の / 平均年収 / を / 教 / えてください / 。
Numbers : [982, 362, 379, 362, 1239, 373, 402, 403, 364]
Decoded : 営業部の社員の平均年収を教えてください。
Same as input: YES  (9 tokens)
```

The demo also includes 「今日は晴れ」: byte tokens, but the text still comes back exactly. Everything is saved to `docs\DAY7_TOKENIZER_DEMO.md`. Try your own sentence too:

```powershell
.venv\Scripts\python.exe scripts\day7_tokenizer_demo.py --text "開発部で年収が一番高い人は誰ですか。"
```

The default demo file is overwritten when you pass `--text`, so run the plain command last if you want the default examples saved.

## Step 6. Read the report and understand three findings

1. **Real size:** train is 2,028,601 tokens. An average question is 13.6 tokens including `<bos>` and `<eos>`, and the longest is 28. These numbers go into Day 11 (model size) and the context-length decision.
2. **"Partial-character tokens"** are pieces of a character the tokenizer never learned, written byte by byte. Train has only 3, and test none. Validation has 503 in 231 questions, caused by just three pieces: **満** (in training it appears only inside 未満), **昇順** and **という** (combinations the training phrasings never use). No character is new; only the combinations are. This is the held-out split doing its job: validation contains wording the model has never seen. These questions still encode and decode exactly, but they will be harder for the model.
3. **Key terms:** of the 279 key terms that occur in the training questions, 273 are 1 to 3 tokens. The 6 others are multi-piece phrases (数が多い順) or words that only occur inside longer words (勤続). This is the same result as Day 6.

## Step 7. Record and commit

Add to `docs\DECISIONS.md`:

```markdown
## D7. Token files (Day 7)

- Each split is one uint16 stream (<bos> question <eos> ...), little-endian, in data/tokens/, with offsets and a meta file
  holding the vocabulary size and the tokenizer SHA-256 (training refuses a mismatched file).
- Real size: train 2,028,601 tokens; average question 13.6 tokens, longest 28 (with <bos>/<eos>).
- Round trip verified on every question and on the files read back from disk.
```

Then commit (VS Code terminal, `llm_home`). `data/tokens/` is git-ignored, because the files can be rebuilt:

```powershell
git status --short
git add src scripts tests docs run_log.md
git status --short
git commit -m "Day 7 home: tokenizer tests, demo, encoded corpus"
git push
```

`data/tokens/` must not appear in the list.

## Talking points for the review

- **Why test every question, not a sample:** a tokenizer that loses even one character silently corrupts the training data. Checking all of them, and the files read back from disk, rules that out.
- **Why one stream of numbers with `<bos>`/`<eos>`:** the trainer cuts fixed-length windows from the stream. The start and end markers tell the model where a question begins and ends, which matters because Model A has to learn to finish a question.
- **Why 16-bit numbers:** our vocabulary (1,674) fits easily, and the files take half the space of 32-bit numbers.
- **Why the meta file holds the tokenizer hash:** if the tokenizer ever changes, training will refuse old token files instead of learning from mismatched numbers.
- **What the partial-character tokens in validation mean:** the held-out phrasings use a few combinations never seen in training (満, 昇順, という). This shows the validation set really tests unseen wording.
- **Task 9 in one sentence:** every piece has a fixed number, and the numbers turn back into exactly the original text. The demo and the tests prove it.

## End of Day 7 checklist

- [ ] Files copied (`src\token_data.py` in `src`, the two scripts, the new test)
- [ ] pytest before: `33 passed, 1 skipped`
- [ ] Encoding: round trip and read-back PASS for all three splits; train 2,028,601 tokens; hashes match
- [ ] pytest after: `34 passed`
- [ ] Demo run; `docs\DAY7_TOKENIZER_DEMO.md` saved
- [ ] D7 added; committed and pushed; `data/tokens/` not in Git

## Send back

1. The console output of Step 3 (as text).
2. The pytest result line of Step 4.
3. When you want it, ask me for the **task 9 document**. Task 9 is complete after today.

## Report snippet (Japanese, for `docs/weekly_reports.md`)

```
【課題9】番号にして元に戻す仕組み
状況: 予定通り終わった
内容: トークンを番号に変換し、番号から元の文に戻す仕組みを確認した。学習用・検証用・テスト用の
      すべての質問文と、絵文字・英語・記号を含む文で、完全に元に戻ることを確認した。質問文を
      番号の列に変換し、学習用のファイルとして保存した。文→トークン→番号→文のデモを作成した。
証拠: docs/DAY7_TOKENIZER_DEMO.md、docs/DAY7_ENCODING_REPORT.md、run_log.md、tests/
次: 課題10(次の単語を予測する仕組み)に向けて、モデルを作る(Day 8)
```
