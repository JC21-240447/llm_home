# Day 7 — day7_encode_corpus.py — Simple Line-by-Line Explanation (English)

> **How the four Day 7 files fit together**
> - **Day 7.1 `token_data.py`** — the file-format library (read it **first**; this script uses it).
> - **Day 7 (this file) `day7_encode_corpus.py`** — the *main script*: tests the tokenizer on **every** question, writes the token files, reads them back, and reports real token counts.
> - **Day 7 `day7_tokenizer_demo.py`** — a small demo script.
> - **Day 7.2 `test_token_data.py`** — automatic tests for the library.

---

## 0. The big picture (read this first)

Day 6 built the tokenizer. Day 7 asks: **"Can we trust it on our whole corpus, and can we store the result safely?"**

Think of a **quality-control stage before shipping**:

| Step | What happens |
|---|---|
| 1 | Load the tokenizer (Day 6) and the three splits (Day 5) |
| 2 | **Round trip on every single question** (and on tricky texts): `decode(encode(text)) == text`. One failure → stop |
| 3 | Measure how many tokens each **key glossary term** needs → write `docs/DAY7_TERM_TOKENS.csv` |
| 4 | Encode each split into one stream `<bos> question <eos> ...` → `data/tokens/{train,val,test}.bin` (+ offsets + meta) |
| 5 | **Read the files back from disk** and decode them: every question must come back exactly |
| 6 | Report the **real** token counts (Day 5 could only give a lower and an upper bound) |

Run it:

```
.venv\Scripts\python.exe scripts\day7_encode_corpus.py --machine home
```

On a Linux server later: `.venv/bin/python scripts/day7_encode_corpus.py --machine server` (forward slashes, Linux venv path). The token files must be **identical** on both machines (compare the SHA-256 values in the report).

---

## 1. Words you need to know (mini dictionary)

| Word | Simple meaning |
|---|---|
| **Round trip** | Encode, then decode, and compare with the original. |
| **Stream** | All questions joined into one long list of IDs, each wrapped as `<bos> … <eos>` (see Day 7.1). |
| **Read-back** | Reading the written file again from disk to prove the file itself is correct (not only the program's memory). |
| **Partial-character token** | A token that is only *part* of a character. It appears for characters the tokenizer never learned (they are written byte by byte). |
| **Key terms** | Column names, values and query words (列名, 値, 検索語), standard terms and their variants, from the glossary (see Day 6). |
| **Tokens per character** | `tokens ÷ characters`. Lower = better compression. |

---

## 2. The docstring

- Same style as before. It lists the 6 steps above, the usage command, and a **server note**: same code on Linux, but use forward slashes and the label `server`.
- Note step 4's format `<bos> question <eos> <bos> question <eos> ...` — exactly what Day 7.1 writes.

---

## 3. Imports and setup

```python
from __future__ import annotations

import argparse
import csv
import hashlib
import platform
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
```
- All **same as Days 4–6** (`Counter` is used for counting, `platform` for the Python version). `REPO_ROOT` and the `sys.path.insert` line are the same trick as Day 6 (so `from src... import` works).

```python
from src.bpe_tokenizer import BOS_ID, BYTE_OFFSET, EOS_ID, ByteBPETokenizer  # noqa: E402
from src.token_data import (  # noqa: E402
    FORMAT_VERSION, TokenFileMeta, encode_lines, read_token_files, sha256_bytes, split_stream, write_token_files,
)
```
- From the tokenizer engine (Day 6.1): the marker IDs, the byte offset (104) and the tokenizer class.
- From the file-format library (**Day 7.1**): the format name, the meta box, and the write/read/split helpers.

---

## 4. Constants

```python
SCRIPT_VERSION: str = "day7-v1"
SPLITS: tuple[str, ...] = ("train", "val", "test")
KEY_DB_GROUPS: frozenset[str] = frozenset({"列名", "値", "検索語"})
```
- `SCRIPT_VERSION` as before; `SPLITS` same as Day 5; `KEY_DB_GROUPS` same as Day 6 (the glossary categories that count as "key").

```python
TRICKY_TEXTS: tuple[str, ...] = (
    "", "今日は晴れ", "Hello, world! 123", "絵文字😀👍と記号①②③※", "改行\nとタブ\tを含む文",
    "ＡＢＣ全角とﾊﾝｶｸ", "<bos><eos><unused_7> は普通の文字", "「」『』【】・〜…", "𠮷野家(4バイトの漢字)",
)
```
- Nine nasty texts, similar to Day 6.2's list: empty text, plain Japanese, English with digits, **emoji** and symbols, a **newline and a tab**, **full-width/half-width** forms, text that **looks like special tokens** (must be ordinary text), Japanese brackets and punctuation, and `𠮷` — a **4-byte kanji** (outside the normal kanji range, so the tokenizer treats it as an "other" character; it must still round-trip).

---

## 5. The `SplitReport` dataclass

```python
@dataclass
class SplitReport:
    split: str
    questions: int
    characters: int
    tokens: int                # including <bos>/<eos>
    content_tokens: int        # without <bos>/<eos>
    avg_tokens: float          # per question, including <bos>/<eos>
    max_tokens: int
    partial_tokens: int        # tokens that are only part of a character (a character never learned)
    questions_with_partial: int
    tokens_sha256: str
```
- One "result card" per split (same idea as Day 6's `SizeResult`): number of questions and characters; total tokens with and without the markers; average and longest question (in tokens); how many partial-character tokens and how many questions contain them; the fingerprint of the `.bin` file.

---

## 6. Helper functions

### 6.1 `read_lines`

```python
def read_lines(path: Path) -> list[str]:
    return [ln for ln in path.read_text(encoding="utf-8").split("\n") if ln]
```
- Same as Day 6.

### 6.2 `load_key_terms`

```python
def load_key_terms(glossary: Path, synonyms: Path) -> list[tuple[str, str, str]]:
    """(term, group, source) for column names, values and query words, standard terms and variants."""
    out: dict[str, tuple[str, str, str]] = {}
    if glossary.exists():
        with glossary.open(encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                g = r["DB区分"].strip()
                t = r["標準用語"].strip()
                if g in KEY_DB_GROUPS and t:
                    out.setdefault(t, (t, g, "用語集"))
    if synonyms.exists():
        with synonyms.open(encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                g = r["DB区分"].strip()
                t = r["変形・同義語"].strip()
                if g in KEY_DB_GROUPS and t:
                    out.setdefault(t, (t, g, "同義語辞書"))
    return sorted(out.values())
```
- Similar to Day 6's version, but it returns **triples** `(term, group, source)` — so the CSV can say where each term came from — and it does **not** filter by the training text.
- `out` is a dictionary keyed by the term. `setdefault` keeps the **first** entry for a term, so a term that is in both files is labelled `用語集` (glossary), because the glossary is read first.
- The glossary block (standard terms) and the synonym block (variants) are the same idea with different column names and the source label.
- `sorted(out.values())` returns the triples sorted (by term first).

### 6.3 `partial_token_ids`

```python
def partial_token_ids(tok: ByteBPETokenizer) -> set[int]:
    """IDs whose bytes are not complete characters: used only for characters training never learned.
    (Single ASCII bytes such as digits are complete characters and are NOT counted.)"""
    out: set[int] = set()
    for i in range(BYTE_OFFSET, tok.vocab_size):
        try:
            tok.token_bytes[i].decode("utf-8")
        except UnicodeDecodeError:
            out.add(i)
    return out
```
- Finds every token ID that is **only part of a character**.
- Loop over all IDs from 104 (the first byte token) to the end of the vocabulary. For each, try to decode its bytes as UTF-8 **on their own**. If that fails (`UnicodeDecodeError`), the token is incomplete → add it to the set.
- Why it works: a single byte like `0xE4` (the start of a 3-byte Japanese character) is not valid UTF-8 alone. A single ASCII byte such as `5` **is** valid, so digits and ASCII symbols are **not** counted.
- Why we care: a learned character normally has a token; a character never seen in training is written **byte by byte** → partial tokens. Counting them tells how often the tokenizer meets "unknown" characters.

### 6.4 `write_report`

```python
def write_report(path, reports, tok, tok_sha, term_rows, n_tricky) -> None:
    in_train = [r for r in term_rows if r[5] == "yes"]
    dist_all: Counter = Counter(min(int(r[3]), 6) for r in term_rows)
    dist_in: Counter = Counter(min(int(r[3]), 6) for r in in_train)
    long_in = [f"{r[0]} ({r[3]})" for r in in_train if int(r[3]) > 3]
```
- Writes `docs/DAY7_ENCODING_REPORT.md`. Each `term_rows` row is `[term, group, source, tokens, pieces, in_train_text]`, so `r[5]` is `"yes"/"no"` and `r[3]` is the token count (text).
- `in_train` — terms that occur in the training questions.
- `dist_all`, `dist_in` — **how many terms need 1, 2, 3, … tokens**. `min(count, 6)` puts everything of 6 or more into one bucket "6".
- `long_in` — terms in training text that need **more than 3** tokens, written like `勤続 (4)`.

The Markdown `lines` contain these sections:

| Section | Content |
|---|---|
| Header | version, date, tokenizer vocabulary and SHA-256 |
| **Round trip** | every question + the tricky texts passed; token files read back OK |
| **Real token counts** | One row per split: questions, characters, tokens (with markers), **tokens per character = `content_tokens / characters`**, average tokens per question, longest question, partial-character tokens, questions using them. Below the table: what "partial-character tokens" means, and that tokens per character excludes `<bos>/<eos>` |
| **Key terms** | How many key terms, how many occur in training; a table "tokens needed → number of terms" (`'6 or more'` label for the last bucket); the list of terms needing more than 3 tokens |
| **Files (SHA-256)** | one fingerprint per `.bin` file |

- `*( ... for r in reports)` and `*( ... for k in sorted(dist_all))` unpack table rows (same trick as before). `dist_in[k]` returns 0 when a bucket is missing (a `Counter` property).
- `path.write_text("\n".join(lines), encoding="utf-8", newline="\n")`: LF line endings, as in Days 5/6.

### 6.5 `append_run_log`

```python
def append_run_log(log_path: Path, machine: str, reports: list[SplitReport], tok_sha: str) -> None:
    body = [
        f"- Machine label: {machine} (CPU only; no GPU used)",
        f"- Script: scripts/day7_encode_corpus.py ({SCRIPT_VERSION}); Python {platform.python_version()}",
        f"- Tokenizer SHA-256: {tok_sha}",
        *(f"- {r.split}: {r.questions:,} questions, {r.tokens:,} tokens (avg {r.avg_tokens:.1f}, max {r.max_tokens}), "
          f"SHA-256 {r.tokens_sha256}" for r in reports),
        "- Checks: round trip PASS on every question and tricky text; disk read-back PASS",
    ]
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"\n## {datetime.now():%Y-%m-%d %H:%M} | {machine} | Day 7 encode corpus\n\n" + "\n".join(body) + "\n")
```
- Same style as Day 6: build the bullet lines (one per split, via the `*( ... )` unpacking), then append a headed entry to `run_log.md`. The "Checks … PASS" line is always written because failures stop the script earlier.

---

## 7. `parse_args`

```python
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Day 7: test the tokenizer on every question and encode the corpus.")
    ap.add_argument("--machine", choices=["home", "server", "school"], required=True)
    ap.add_argument("--tokenizer", type=Path, default=REPO_ROOT / "data/tokenizer/tokenizer.json")
    ap.add_argument("--corpus-dir", type=Path, default=REPO_ROOT / "data/corpus")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data/tokens")
    ap.add_argument("--glossary", type=Path, default=REPO_ROOT / "data/source/用語集.csv")
    ap.add_argument("--synonyms", type=Path, default=REPO_ROOT / "data/source/同義語辞書.csv")
    ap.add_argument("--report", type=Path, default=REPO_ROOT / "docs/DAY7_ENCODING_REPORT.md")
    ap.add_argument("--terms-csv", type=Path, default=REPO_ROOT / "docs/DAY7_TERM_TOKENS.csv")
    ap.add_argument("--run-log", type=Path, default=REPO_ROOT / "run_log.md")
    ap.add_argument("--no-log", action="store_true")
    return ap.parse_args()
```
- Same `argparse` idea as before. New: `--machine` now also accepts **`server`** (the future Linux machine). The `Path` options (tokenizer, corpus dir, output dir, glossary, synonyms, report, terms CSV, run log) all work alike. `--no-log` as before.

---

## 8. `main`

### 8.1 Setup and loading

```python
def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    a = parse_args()
    for name in ("tokenizer", "corpus_dir", "out_dir", "glossary", "synonyms", "report", "terms_csv", "run_log"):
        setattr(a, name, getattr(a, name).resolve())
    if not a.tokenizer.exists():
        raise SystemExit(f"Missing {a.tokenizer}. Run Day 6 first.")
```
- Same prelude as Days 4–6 (console safety, absolute paths). The tokenizer file must exist (from Day 6).

```python
    print(f"=== Day 7 encode corpus | machine={a.machine} | Python {platform.python_version()} ===")
    tok = ByteBPETokenizer.load(a.tokenizer)
    tok_sha = sha256_bytes(a.tokenizer.read_bytes())
    print(f"Tokenizer: vocabulary {tok.vocab_size:,}, SHA-256 {tok_sha[:12]}...")
```
- Print the header, load the tokenizer, compute its fingerprint (`tok_sha`; it goes into every meta file), and print the first 12 characters (`[:12]`) as a short form.

### 8.2 Round trip on tricky texts

```python
    bad = [t for t in TRICKY_TEXTS if tok.decode(tok.encode(t)) != t]
    if bad:
        raise SystemExit(f"Round trip FAILED on tricky texts: {bad}")
    print(f"Round trip on {len(TRICKY_TEXTS)} tricky texts: PASS")
```
- Collect the tricky texts that do **not** survive encode→decode. If any → stop. Otherwise PASS.

### 8.3 For every split

```python
    partial_ids = partial_token_ids(tok)
    reports: list[SplitReport] = []
    for split in SPLITS:
        path = a.corpus_dir / f"{split}.txt"
        if not path.exists():
            raise SystemExit(f"Missing {path}. Run Day 5 first.")
        lines = read_lines(path)
```
- Compute the set of partial-character IDs once. Then, for `train`, `val`, `test`: find the text file (from Day 5; stop if missing) and read its lines.

```python
        unique = list(dict.fromkeys(lines))
        failures = [ln for ln in unique if tok.decode(tok.encode(ln)) != ln]
        if failures:
            raise SystemExit(f"Round trip FAILED on {len(failures)} {split} questions, e.g. {failures[0]!r}")
```
- `dict.fromkeys(lines)` makes a dictionary whose keys are the lines → **removes repeats while keeping the order**. (Train contains repeated lines from oversampling, Day 5; testing each different sentence once is enough and faster.)
- Check **every unique question** round-trips. Any failure → stop, showing how many and the first example (`!r` shows quotes).

```python
        tokens, offsets = encode_lines(lines, tok.encode, BOS_ID, EOS_ID)
        meta = TokenFileMeta(FORMAT_VERSION, split, tok.vocab_size, tok_sha, BOS_ID, EOS_ID, len(lines), len(tokens),
                             sha256_bytes(tokens.tobytes()), sha256_bytes(path.read_bytes()))
        write_token_files(a.out_dir, split, tokens, offsets, meta)
```
- Encode **all** lines (including repeats, because the stream must contain every training line) into `tokens` and `offsets` (Day 7.1).
- Build the meta "label" with its fields **in order**: format, split, vocabulary size, tokenizer fingerprint, `<bos>`/`<eos>` IDs, number of questions, number of tokens, fingerprint of the token bytes, fingerprint of the source text file.
- Write the three files for this split.

```python
        disk_tokens, disk_offsets, _ = read_token_files(a.out_dir, split, memmap=True)
        for ln, ids in zip(lines, split_stream(disk_tokens, disk_offsets)):
            if ids[0] != BOS_ID or ids[-1] != EOS_ID or tok.decode(ids.tolist()) != ln:
                raise SystemExit(f"Disk read-back FAILED in {split}: {ln!r}")
```
- **Read-back from disk** (as a memory map): open the files we just wrote, cut the stream back into questions, and for each original line check that the question starts with `<bos>`, ends with `<eos>`, and decodes to **exactly** the original line. This proves the **files** are right, not just the in-memory arrays. (`ids.tolist()` converts the NumPy array to normal Python ints for `decode`.) The `_` ignores the returned meta.

```python
        per_q = [int(e - s) for s, e in zip(offsets, list(offsets[1:]) + [len(tokens)])]
        is_partial = [int(x) in partial_ids for x in tokens.tolist()]
        q_partial = sum(1 for ids in split_stream(tokens, offsets) if any(int(x) in partial_ids for x in ids.tolist()))
```
- `per_q` — tokens per question (including markers) = end position minus start position, same "next start is my end" idea as `split_stream`.
- `is_partial` — for **every token**, is it a partial-character token? (a list of True/False; summing it counts them.)
- `q_partial` — in how many **questions** does at least one partial token appear (`any(...)` = "at least one").

```python
        r = SplitReport(split, len(lines), sum(len(ln) for ln in lines), len(tokens), len(tokens) - 2 * len(lines),
                        sum(per_q) / len(per_q), max(per_q), sum(is_partial), q_partial,
                        hashlib.sha256((a.out_dir / f"{split}.bin").read_bytes()).hexdigest())
        reports.append(r)
```
- Make the result card: questions; total characters; total tokens; **content tokens = tokens − 2 per question** (remove `<bos>` and `<eos>`); average and longest question; partial token count; questions with partial; and the SHA-256 of the **file on disk**.

```python
        print(f"{split:<6} {r.questions:>8,} questions  {r.tokens:>10,} tokens  avg {r.avg_tokens:4.1f}  max {r.max_tokens:>3}  "
              f"partial-char {r.partial_tokens:>4} in {r.questions_with_partial} questions  -> round trip + read-back PASS")
```
- Print one neat summary line for the split (alignment codes as in Day 5/6). Reaching this line means both checks passed.

### 8.4 Key terms

```python
    terms = load_key_terms(a.glossary, a.synonyms)
    train_blob = "\n".join(read_lines(a.corpus_dir / "train.txt"))
    term_rows: list[list[str]] = []
    for term, group, source in terms:
        n = len(tok.encode(term))
        term_rows.append([term, group, source, str(n), " / ".join(tok.pieces(term)), "yes" if term in train_blob else "no"])
```
- Load the key terms; make one big string of the training text (as in Day 6).
- For each term: `n` = number of tokens needed; a row with the term, group, source, `n` (as text), the readable pieces joined with ` / `, and whether the term **appears anywhere** in the training text (`term in train_blob` — a substring check).

```python
    a.terms_csv.parent.mkdir(parents=True, exist_ok=True)
    with a.terms_csv.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["term", "DB区分", "source", "tokens", "pieces", "in_train_text"])
        w.writerows(term_rows)
```
- Save the table as a CSV (`utf-8-sig` so Excel shows Japanese correctly; `newline=""` for the `csv` module — same as Day 5's label file). `writerow` writes the header; `writerows` writes all rows at once.

```python
    in_train = [r for r in term_rows if r[5] == "yes"]
    print(f"Key terms: {len(term_rows)} (in training text: {len(in_train)}); "
          f"1-3 tokens: {sum(1 for r in term_rows if int(r[3]) <= 3)} of all, "
          f"{sum(1 for r in in_train if int(r[3]) <= 3)} of those in the training text")
```
- Print how many key terms there are, how many occur in training, and how many are 1–3 tokens (overall and among those in training).

### 8.5 Report, log, exit

```python
    write_report(a.report, reports, tok, tok_sha, term_rows, len(TRICKY_TEXTS))
    total = sum(r.tokens for r in reports)
    print(f"\nTotal tokens: {total:,} (train {reports[0].tokens:,})")
    print(f"Report: {a.report}\nTerm table: {a.terms_csv}")
    if not a.no_log:
        append_run_log(a.run_log, a.machine, reports, tok_sha)
        print(f"Appended an entry to {a.run_log.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```
- Write the report, print total tokens (train is `reports[0]` because `SPLITS` starts with train), print the file locations, append the run log unless `--no-log`, return 0. Same ending as before.

---

## 9. The whole run in order

1. Options → absolute paths → tokenizer must exist.
2. Load tokenizer, fingerprint it.
3. Round trip on 9 tricky texts.
4. For each split: round trip on **every unique question** → encode → write 3 files → **read back and decode every line** → count tokens/partial tokens → fingerprint.
5. Key terms: tokens per term → CSV.
6. Report → run log.

## 10. How to read the results

- **All PASS lines** = the tokenizer is lossless on the whole corpus and the files on disk are correct.
- **Real tokens vs Day 5 bounds:** the real total for each split should be between Day 5's "token lower bound" and its "UTF-8 bytes".
- **Tokens per character** (e.g. 0.4) tells the compression; lower is better.
- **Longest question (tokens)** feeds the context-length decision (Day 6 also gave an estimate; this one is exact because it covers every line).
- **Partial-character tokens:** ideally 0 or very small in train. In val/test, a few are normal (characters that only occur there).
- **Key-term table:** terms needing more than 3 tokens are phrases of several pieces or words that only occur inside longer words (as in Day 6).
- **SHA-256 of the `.bin` files:** compare between home and server — must be identical.

## 11. Small things worth knowing

- **Key-term "in training" is a substring check** (`term in train_blob`), so a short term can match inside a longer word. Day 6 used a stricter "achievable" test.
- **Empty split files** would crash the average (`sum(per_q) / len(per_q)` divides by zero). With real Day 5 output this does not happen.
- **Files are written before the read-back check.** If read-back fails, the files already exist; treat them as invalid.
- **Train round trip is on unique lines, the read-back is on all lines.** Both are complete: repeats are identical to their originals.
- **`memmap=True` read-back** keeps a file handle open on Windows until the arrays go out of scope; harmless here because nothing deletes the files during the run.

## 12. Quick summary in one paragraph

The script proves the tokenizer is lossless on every question of train, validation and test (plus nasty texts), writes each split as a `<bos> … <eos>` stream of `uint16` token IDs with offsets and a meta file, re-reads the files from disk to prove they decode back exactly, measures real token counts, partial-character tokens and key-term token needs, and records everything with SHA-256 fingerprints in a report, a CSV and the run log.
