# Day 6 — day6_train_bpe.py — Simple Line-by-Line Explanation (English)

> **How the three Day 6 files fit together**
> - **Day 6.1 `bpe_tokenizer.py`** — the *engine* (the tokenizer itself).
> - **Day 6 (this file) `day6_train_bpe.py`** — the *driver*: trains the engine, tries several sizes, picks one by a written rule, saves, verifies, reports.
> - **Day 6.2 `test_bpe_tokenizer.py`** — the *inspector* (automatic tests).
>
> Read **Day 6.1 first** — this script constantly uses its functions (`train_bpe`, `count_pieces`, `pretokenize`, `ByteBPETokenizer`).

---

## 0. The big picture (read this first)

The big question of Day 6: **how big should the tokenizer's vocabulary be?**

- **Too small** → words are cut into many tiny pieces → long token sequences, slow, harder for the model.
- **Too large** → many tokens are so rare that the model barely learns them, and the model's embedding table gets bigger for no gain.

Instead of guessing, this script **measures** and then chooses with a **written rule**. The clever trick (from Day 6.1): merges are learned in order, so **the first N merges are the size-N tokenizer**. One training run therefore gives us *every* smaller size for free.

| Step | What happens |
|---|---|
| 1 | Read `train.txt` (for training) and `val.txt` (for measuring only) |
| 2 | Train once, learning every merge the text allows → "full" tokenizer |
| 3 | Cut it down to many candidate sizes (400, 500, ..., 16000) and **measure** each |
| 4 | Apply the rule `choose_size()` → pick the smallest good size |
| 5 | Save the chosen tokenizer + the full one |
| 6 | Reload and **verify** (same behaviour, lossless round trip) |
| 7 | Print demos, write a report and a run-log entry |

Run it:

```
.venv\Scripts\python.exe scripts\day6_train_bpe.py --machine home
```

Later, to save a different size **without training again**:

```
.venv\Scripts\python.exe scripts\day6_train_bpe.py --machine home --resave-size 4000
```

---

## 1. Words you need to know (mini dictionary)

| Word | Simple meaning |
|---|---|
| **Vocabulary size** | How many different tokens the tokenizer has. |
| **Compression** | How few tokens are needed for the same text. Fewer = better compression. |
| **Tokens per character** | `tokens ÷ characters`. Lower = better compression. |
| **Key terms** | Important words: column names, values and query words (列名, 値, 検索語) from the glossary. We want each to be only 1–3 tokens. |
| **Achievable key term** | A key term that training *can* turn into 1–3 tokens (see section 6.4). |
| **Context length** | The maximum number of tokens the model reads at once. The longest question (in tokens) helps decide it. |
| **Tolerance** | How much worse than the best (largest) tokenizer we accept (default 5%). |
| **Truncate** | Keep only the first N merges (Day 6.1 `truncated`). |
| **Held-out validation** | Text not used for training (Day 5). Here it only *measures*. |
| **Round trip** | `decode(encode(text)) == text`. |

---

## 2. The docstring (top of the file)

- Same style as before. It lists the 5 steps and the **rule for choosing the size**:
  > the smallest candidate where **every key term that CAN be 1–3 tokens is 1–3 tokens**, and **validation needs at most `--tolerance` more tokens** than the largest tokenizer.
- Important: validation and test text are **not** used for training. Validation is only used to measure the candidates.
- School-PC note: same code, CPU only; the school PC must produce the **same `tokenizer.json` SHA-256**.

---

## 3. Imports and the `sys.path` line

```python
from __future__ import annotations

import argparse
import csv
import hashlib
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
```
- All **same as Days 4 and 5**, except the new `platform` (gives the Python version, e.g. `platform.python_version()`).

```python
REPO_ROOT: Path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
```
- `REPO_ROOT` — same as before.
- `sys.path` is the list of folders Python searches when you `import`. `insert(0, ...)` puts the repo root **first**, so `from src.bpe_tokenizer import ...` works even though this script lives in `scripts/`.

```python
from src.bpe_tokenizer import (  # noqa: E402
    FIRST_MERGE_ID,
    ByteBPETokenizer,
    TrainingLog,
    count_pieces,
    pretokenize,
    train_bpe,
)
```
- Imports the engine from Day 6.1. `# noqa: E402` tells the style checker "yes, this import is not at the top of the file — on purpose, because `sys.path` had to be set first."

---

## 4. Constants

```python
SCRIPT_VERSION: str = "day6-v1"
KEY_DB_GROUPS: frozenset[str] = frozenset({"列名", "値", "検索語"})
```
- `SCRIPT_VERSION` — as before.
- `KEY_DB_GROUPS` — the glossary's `DB区分` (category) values that count as **key**: column names, values, query words. A `frozenset` can't be changed.

```python
CANDIDATE_SIZES: tuple[int, ...] = (400, 500, 600, 800, 1000, 1200, 1400, 1500, 1600, 2000, 4000, 8000, 16000)
```
- The vocabulary sizes we will try (in ascending order — `choose_size` relies on this order).

```python
DEMO_SENTENCES: tuple[str, ...] = (
    "今日は晴れ",
    "営業部の社員の平均年収を教えてください。",
    "ボーナスが100万円以上の人は何人ですか。",
    "大阪支社で勤続年数が一番長い人は誰？",
    "Model A tokenizer test 2026",
)
```
- Sample sentences shown at the end, so you can *see* how the chosen tokenizer cuts real text (including an English one with a number).

---

## 5. The `SizeResult` dataclass

```python
@dataclass
class SizeResult:
    """Measurements for one candidate vocabulary size."""

    vocab_size: int
    val_tokens: int
    tokens_per_char: float
    key_ok: int          # achievable key terms that are 1-3 tokens
    key_achievable: int  # key terms that training can make 1-3 tokens (see achievable_terms())
    max_val_tokens: int
    max_train_tokens: int
```
- One "row of results" for one candidate size:
  - `vocab_size` — the size measured.
  - `val_tokens` — total tokens needed for all validation text (smaller = better).
  - `tokens_per_char` — the same, divided by the number of characters.
  - `key_ok` — how many achievable key terms are ≤ 3 tokens at this size.
  - `key_achievable` — how many key terms *could* be (the target).
  - `max_val_tokens`, `max_train_tokens` — the longest question in tokens (validation, and a sample of train).

---

## 6. Helper functions

### 6.1 `sha256_file` and `read_lines`

```python
def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_lines(path: Path) -> list[str]:
    return [ln for ln in path.read_text(encoding="utf-8").split("\n") if ln]
```
- `sha256_file` — same as Day 4.
- `read_lines` — read the text file, split at line breaks, and keep only non-empty lines (`if ln`).

### 6.2 `load_key_terms`

```python
def load_key_terms(glossary: Path, synonyms: Path, train_blob: str) -> list[str]:
    """Key terms (列名, 値, 検索語 and their variants) that actually occur in the training text."""
    terms: set[str] = set()
    if glossary.exists():
        with glossary.open(encoding="utf-8-sig", newline="") as f:
            terms |= {r["標準用語"].strip() for r in csv.DictReader(f) if r["DB区分"].strip() in KEY_DB_GROUPS}
    if synonyms.exists():
        with synonyms.open(encoding="utf-8-sig", newline="") as f:
            terms |= {r["変形・同義語"].strip() for r in csv.DictReader(f) if r["DB区分"].strip() in KEY_DB_GROUPS}
    return sorted(t for t in terms if t and t in train_blob)
```
- Start with an empty set.
- From the glossary (if it exists): take the `標準用語` of every row whose `DB区分` is one of the key groups. `terms |= {...}` **adds** the new set to `terms` (set union).
- From the synonym dictionary: the same, using the `変形・同義語` (variant) column.
- Return the terms (sorted) that are non-empty **and actually appear in the training text** (`t in train_blob`). Terms never seen in training cannot be learned, so measuring them would be pointless.

### 6.3 `achievable_terms`

```python
def achievable_terms(key_terms: list[str], piece_counts: dict[str, int], min_frequency: int) -> list[str]:
    out: list[str] = []
    for t in key_terms:
        parts = pretokenize(t)
        if len(parts) <= 3 and all(piece_counts.get(p, 0) >= min_frequency for p in parts):
            out.append(t)
    return out
```
- Which key terms *can* realistically become 1–3 tokens?
- `parts = pretokenize(t)` — cut the term into pre-token pieces (Day 6.1).
- It is achievable if **both**:
  1. It has **at most 3 pieces**, and
  2. **every piece occurs on its own** in the training text at least `min_frequency` times (`piece_counts.get(p, 0)`; `all(...)` = "true for every piece").
- Why? BPE can never merge across a piece boundary, so each piece needs at least one token → ≤ 3 pieces gives ≤ 3 tokens. And a piece can only become **one** token if it occurs often enough as a whole piece. The docstring's example: `勤続` appears only *inside* `勤続年数` (one kanji run), so `勤続` alone is **not** a piece in the training text → it can't become its own token → not achievable. This keeps the rule fair: we don't demand the impossible.

### 6.4 `measure`

```python
def measure(tok, val, train_sample, key_terms, achievable) -> SizeResult:
    val_lengths = [len(tok.encode(ln)) for ln in val]
    key_ok = sum(1 for t in achievable if len(tok.encode(t)) <= 3)
    chars = sum(len(ln) for ln in val)
    return SizeResult(
        vocab_size=tok.vocab_size,
        val_tokens=sum(val_lengths),
        tokens_per_char=sum(val_lengths) / chars,
        key_ok=key_ok,
        key_achievable=len(achievable),
        max_val_tokens=max(val_lengths),
        max_train_tokens=max(len(tok.encode(ln)) for ln in train_sample),
    )
```
- `val_lengths` — the token count of every validation line.
- `key_ok` — how many achievable key terms are encoded in ≤ 3 tokens.
- `chars` — total characters of validation text.
- Return a `SizeResult`: the size, total validation tokens, tokens per character, key-term counts, and the longest question (validation; train sample).
- (Note: the `key_terms` parameter is passed in but not used inside — only `achievable` is. Harmless.)

### 6.5 `choose_size` — the written rule

```python
def choose_size(results: list[SizeResult], tolerance: float) -> SizeResult:
    best_val = results[-1].val_tokens
    for r in results:
        if r.key_ok == r.key_achievable and r.val_tokens <= (1 + tolerance) * best_val:
            return r
    return results[-1]
```
- `results[-1]` is the **largest** tokenizer (results are in ascending order), so `best_val` = the fewest validation tokens we can ever get.
- Go from the smallest size upward. Return the **first** one where:
  - `key_ok == key_achievable` (every achievable key term is 1–3 tokens), **and**
  - `val_tokens <= (1 + tolerance) * best_val` (within 5% of the best compression).
- If none qualifies, fall back to the largest.
- The rule is **written down in code** and in the report, so the choice is explainable ("smallest that satisfies the evidence"), not a feeling.

### 6.6 `roundtrip_failures`

```python
def roundtrip_failures(tok: ByteBPETokenizer, lines: list[str]) -> int:
    return sum(1 for ln in lines if tok.decode(tok.encode(ln)) != ln)
```
- Count lines where encode→decode does **not** give back the original. Should be **0**.

### 6.7 `write_report`

```python
def write_report(path, log, results, chosen, tolerance, min_frequency, seconds, demos,
                 key_terms, achievable, tok, hashes, n_train, n_val, new_val_pieces) -> None:
    long_terms = [t for t in achievable if len(tok.encode(t)) > 3]
    short_all = sum(1 for t in key_terms if len(tok.encode(t)) <= 3)
    long_other = [f"{t} ({len(tok.encode(t))})" for t in key_terms if t not in achievable and len(tok.encode(t)) > 3]
```
- Writes `docs/DAY6_TOKENIZER_REPORT.md`. First it prepares three lists:
  - `long_terms` — achievable key terms that are still **more than 3 tokens** at the chosen size (should be none).
  - `short_all` — how many of **all** key terms are ≤ 3 tokens.
  - `long_other` — non-achievable key terms longer than 3 tokens, written like `勤続 (2)` with their token count.

The Markdown `lines` list contains these sections:

| Section | Content |
|---|---|
| Title + date | version and time |
| **Training** | training line count, validation line count; if `log` exists (not in `--resave-size` mode): pieces total/unique, upper bound on merges, **merges learned**, **largest possible vocabulary** (`FIRST_MERGE_ID + merges_learned`), the stop reason, time |
| **Size candidates** | A table, one row per size: vocabulary, validation tokens, tokens per character, key terms at 1–3 tokens (`ok/achievable`), longest validation question, longest train question. The chosen row is marked `**(chosen)**` (`r is chosen` compares the **same object**) |
| **Rule** | The rule in words with the tolerance as a percent (`{tolerance:.0%}`) and the chosen size |
| **Key terms explanation** | how many key terms, how many achievable, which are too long, which cannot be 1–3 tokens and why |
| **New validation pieces** | Pieces in validation that **never occur in training** (they are still encodable from smaller tokens) |
| **Demo** | table of the demo sentences and how they were cut (`' / '.join(p)`) |
| **Files (SHA-256)** | fingerprints of the tokenizer files and the training input |

- `*( ... for r in results)` unpacks one table row per size (same trick as Days 4/5).
- `{' **(chosen)**' if r is chosen else ''}` — adds the mark only to the chosen row.
- At the end: `path.write_text("\n".join(lines), encoding="utf-8")` — same as before.

### 6.8 `append_run_log`

```python
def append_run_log(log_path: Path, machine: str, text: str) -> None:
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"\n## {datetime.now():%Y-%m-%d %H:%M} | {machine} | Day 6 tokenizer\n\n{text}\n")
```
- Simpler than Days 4/5: the caller builds the body text; this just adds a heading and appends it to `run_log.md` (append mode `"a"`).

---

## 7. `parse_args`

```python
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Day 6: train the byte-level BPE tokenizer and choose its size.")
    ap.add_argument("--machine", choices=["school", "home"], required=True)
    ap.add_argument("--train", type=Path, default=REPO_ROOT / "data/corpus/train.txt")
    ap.add_argument("--val", type=Path, default=REPO_ROOT / "data/corpus/val.txt")
    ap.add_argument("--glossary", type=Path, default=REPO_ROOT / "data/source/用語集.csv")
    ap.add_argument("--synonyms", type=Path, default=REPO_ROOT / "data/source/同義語辞書.csv")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data/tokenizer")
    ap.add_argument("--report", type=Path, default=REPO_ROOT / "docs/DAY6_TOKENIZER_REPORT.md")
    ap.add_argument("--run-log", type=Path, default=REPO_ROOT / "run_log.md")
    ap.add_argument("--min-frequency", type=int, default=2)
    ap.add_argument("--max-merges", type=int, default=100_000, help="safety cap; training normally stops earlier")
    ap.add_argument("--tolerance", type=float, default=0.05)
    ap.add_argument("--resave-size", type=int, default=None,
                    help="skip training: cut tokenizer_full.json to this size and save it as tokenizer.json")
    ap.add_argument("--no-log", action="store_true")
    return ap.parse_args()
```
- Same `argparse` idea as before. `--machine` required. The seven `Path` options all work alike (name + default location).
- `--min-frequency` — a pair must occur at least this often to be merged (default 2).
- `--max-merges` — safety cap. `100_000` is just `100000` (underscores make big numbers readable).
- `--tolerance` — 0.05 = 5%.
- `--resave-size` — default `None` (= normal training). If given: **skip training** and re-cut the saved full tokenizer to that size.
- `--no-log` — flag, as in Days 4/5.

---

## 8. `main`

### 8.1 Setup

```python
def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    a = parse_args()
    for name in ("train", "val", "glossary", "synonyms", "out_dir", "report", "run_log"):
        setattr(a, name, getattr(a, name).resolve())
    for p in (a.train, a.val):
        if not p.exists():
            raise SystemExit(f"Missing {p}. Run Day 5 first.")
```
- Same console and `resolve()` pattern as Days 4/5. The two input files must exist (they come from Day 5), else stop.

```python
    print(f"=== Day 6 tokenizer | machine={a.machine} | Python {platform.python_version()} ===")
    train = read_lines(a.train)
    val = read_lines(a.val)
    full_path = a.out_dir / "tokenizer_full.json"
    chosen_path = a.out_dir / "tokenizer.json"
```
- Header (with the Python version), read both text files, and define the two output file paths: the **full** tokenizer and the **chosen-size** tokenizer.

### 8.2 Train (or load)

```python
    log: TrainingLog | None = None
    seconds = 0.0
    if a.resave_size is None:
        start = time.perf_counter()
        merges, log = train_bpe(count_pieces(train), a.max_merges, a.min_frequency)
        seconds = time.perf_counter() - start
        full = ByteBPETokenizer(merges)
        full.save(full_path)
        print(f"Unique pre-token pieces: {log.unique_pieces:,} (total {log.total_pieces:,})")
        print(f"Merges learned: {log.merges_learned:,} -> largest vocabulary {full.vocab_size:,} "
              f"({log.stop_reason}); {seconds:.1f} s")
    else:
        if not full_path.exists():
            raise SystemExit(f"--resave-size needs {full_path}. Run without it first.")
        full = ByteBPETokenizer.load(full_path)
        print(f"Loaded {full_path.name}: largest vocabulary {full.vocab_size:,}")
```
- `log` and `seconds` start empty (the `--resave-size` path has no training log).
- **Normal mode:** start the stopwatch; `count_pieces(train)` counts pieces and `train_bpe` learns the merges (all of them, until it stops itself); stop the stopwatch; build the **full** tokenizer and save it; print the pieces count and the largest vocabulary with the stop reason.
- **Resave mode:** the full file must already exist (else stop with a clear message); load it.

### 8.3 Preparing the measurements

```python
    train_blob = "\n".join(train)
    piece_counts = count_pieces(train)
    key_terms = load_key_terms(a.glossary, a.synonyms, train_blob)
    achievable = achievable_terms(key_terms, piece_counts, a.min_frequency)
    new_val_pieces = sorted({p for ln in val for p in pretokenize(ln)} - set(piece_counts))
    train_sample = train[::10]  # every 10th line: enough to find the longest questions quickly
```
- `train_blob` — all training text as one string (for `t in train_blob`).
- `piece_counts` — piece counts again (also needed in resave mode; in normal mode this recounts, harmless).
- `key_terms` and `achievable` — the targets for the rule.
- `new_val_pieces` — set subtraction: pieces in validation **minus** pieces seen in training = pieces never seen in training.
- `train[::10]` — slice with a step: every 10th line. Used only for the "longest train question" column (see notes).

```python
    sizes = [s for s in CANDIDATE_SIZES if s < full.vocab_size] + [full.vocab_size]
    if a.resave_size is not None and a.resave_size not in sizes:
        sizes = sorted(set(sizes) | {min(a.resave_size, full.vocab_size)})
```
- Candidate sizes smaller than the full size, **plus the full size itself** (so the largest is always measured).
- In resave mode, if your requested size is not in the list, add it (never above the full size).

### 8.4 Measuring each size

```python
    print(f"Key terms found in the training text: {len(key_terms)}; achievable as 1-3 tokens: {len(achievable)}")
    print(f"\n{'vocab':>7} {'val tokens':>11} {'tok/char':>9} {'key terms 1-3':>14} {'max val':>8} {'max train':>10}")
    results: list[SizeResult] = []
    for s in sizes:
        r = measure(full.truncated(s), val, train_sample, key_terms, achievable)
        results.append(r)
        print(f"{r.vocab_size:>7,} {r.val_tokens:>11,} {r.tokens_per_char:>9.3f} "
              f"{f'{r.key_ok}/{r.key_achievable}':>14} {r.max_val_tokens:>8} {r.max_train_tokens:>10}")
```
- Print counts and a table header (alignment codes `>7` etc. as in Day 5).
- For each size: **`full.truncated(s)`** gives the size-`s` tokenizer; `measure` evaluates it; store and print the row. The nested f-string builds text like `35/38` and right-aligns it.

### 8.5 Choosing

```python
    if a.resave_size is None:
        chosen = choose_size(results, a.tolerance)
        print(f"\nRule: smallest size with all achievable key terms at 1-3 tokens and validation within "
              f"{a.tolerance:.0%} of the largest -> chosen {chosen.vocab_size:,}")
    else:
        chosen = next(r for r in results if r.vocab_size == min(a.resave_size, full.vocab_size))
        print(f"\nManual choice (--resave-size): {chosen.vocab_size:,}")
```
- Normal mode: apply the written rule and print it.
- Resave mode: **you** choose; `next(generator)` returns the first result whose size matches.

```python
    tok = full.truncated(chosen.vocab_size)
    tok.save(chosen_path)
```
- Build the chosen tokenizer and save it as `tokenizer.json`.

### 8.6 Verify

```python
    reloaded = ByteBPETokenizer.load(chosen_path)
    reloaded_full = ByteBPETokenizer.load(full_path)
    same = reloaded.merges == tok.merges and reloaded_full.merges == full.merges
    same = same and all(reloaded.encode(ln) == tok.encode(ln) for ln in val)
    extra = ["", "今日は晴れ", "ABC abc 123", "絵文字😀と記号①②", "\t改行\nを含む", "<bos>は普通の文字"]
    failures = roundtrip_failures(reloaded, val + extra)
    if not same or failures:
        raise SystemExit(f"Reload check FAILED: identical={same}, round-trip failures={failures}")
    print(f"Reload check: PASS (identical merges and encodings); round trip: PASS on {len(val) + len(extra):,} texts")
```
- **Reload** both saved files from disk (this proves that saving and loading lose nothing).
- `same` — the reloaded merges equal the originals, **and** every validation line encodes identically before and after reload.
- `extra` — nasty test texts: empty string, English + digits, emoji and circled digits, tab and newline, and a literal `<bos>` (must be treated as ordinary text).
- `failures` — count of round-trip failures on validation lines + extras (should be 0).
- If anything is wrong → stop with an error. Otherwise print PASS.

### 8.7 Demos, hashes, report

```python
    demos = [(t, reloaded.pieces(t)) for t in DEMO_SENTENCES]
    print("\nDemo (chosen tokenizer):")
    for t, p in demos:
        print(f"  {t}\n    -> {' / '.join(p)}  ({len(p)} tokens)")
```
- For each demo sentence, get readable pieces and print them joined with ` / ` plus the token count.

```python
    hashes = {
        "data/tokenizer/tokenizer.json": hashlib.sha256(chosen_path.read_bytes()).hexdigest(),
        "data/tokenizer/tokenizer_full.json": hashlib.sha256(full_path.read_bytes()).hexdigest(),
        "data/corpus/train.txt (input)": sha256_file(a.train),
    }
    a.report.parent.mkdir(parents=True, exist_ok=True)
    write_report(a.report, log, results, chosen, a.tolerance, a.min_frequency, seconds, demos, key_terms,
                 achievable, reloaded, hashes, len(train), len(val), new_val_pieces)
    print(f"\ntokenizer.json SHA-256: {hashes['data/tokenizer/tokenizer.json']}")
    print(f"Report: {a.report}")
```
- Fingerprints of both tokenizer files and the training input (the first two use `hashlib` directly, the third uses `sha256_file` — same result type). **Compare `tokenizer.json`'s SHA-256 between school and home** — it must match.
- Write the report and print the key fingerprint and location.

### 8.8 Log and exit

```python
    if not a.no_log:
        text = "\n".join([
            "- Machine label: " + a.machine + " (CPU only; no GPU used)",
            f"- Script: scripts/day6_train_bpe.py ({SCRIPT_VERSION}); Python {platform.python_version()}",
            (f"- Training: {log.unique_pieces:,} unique pieces, {log.merges_learned:,} merges, {seconds:.1f} s, "
             f"largest vocabulary {full.vocab_size:,}" if log else "- Re-saved from tokenizer_full.json (no training)"),
            f"- Chosen vocabulary: {chosen.vocab_size:,}; validation {chosen.val_tokens:,} tokens "
            f"({chosen.tokens_per_char:.3f} per char); key terms 1-3 tokens {chosen.key_ok}/{chosen.key_achievable}; "
            f"longest question {max(chosen.max_val_tokens, chosen.max_train_tokens)} tokens",
            f"- tokenizer.json SHA-256: {hashes['data/tokenizer/tokenizer.json']}",
            "- Checks: reload PASS, round trip PASS",
        ])
        append_run_log(a.run_log, a.machine, text)
        print(f"Appended an entry to {a.run_log.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```
- Build the log text from bullet lines. The third bullet uses a **conditional expression**: if `log` exists show the training stats, otherwise say "Re-saved ... (no training)".
- "Longest question" = the larger of the longest validation and sampled-train question.
- `"Checks: ... PASS"` is always written because failures stop the program earlier.
- Same ending as Days 4/5.

---

## 9. The whole run in order

1. Read options → paths → check Day 5 files exist.
2. Read train and val lines.
3. Train once (or load the full tokenizer in `--resave-size` mode); save `tokenizer_full.json`.
4. Find key terms → which are achievable.
5. For every candidate size: cut the full tokenizer to that size and **measure**.
6. Choose the size with the rule (or use your manual size).
7. Save `tokenizer.json`; reload both files; verify identity and round trips.
8. Demos → fingerprints → report → run log.

## 10. How to read the results

- **Table:** as the size grows, `val tokens` falls (better compression) but with diminishing returns. The chosen row is the first size that is **good enough on both criteria**.
- **`key terms 1-3` = `ok/achievable`:** the chosen size must show equal numbers (e.g. `38/38`).
- **`max val / max train`:** the longest question in tokens → input for the context-length decision (leave some margin and room for `<bos>`/`<eos>`).
- **`Reload check: PASS`:** saving/loading is lossless and the tokenizer is lossless on all validation lines and tricky texts.
- **`tokenizer.json SHA-256`:** compare with the other PC. Same hash = identical tokenizer.
- **"Validation pieces never seen in training":** normal for held-out patterns; they are still encoded from smaller tokens.

## 11. Small things worth knowing

- **"Longest train question" is a sample.** `max_train_tokens` is computed only on every 10th training line (`train[::10]`), so the true longest training question could be a bit longer. When deciding the context length, either add margin or compute the maximum over all lines (it is fast).
- **Verification runs after saving.** The chosen tokenizer is written to disk *before* the reload check. If the check fails, the files already exist; treat them as invalid.
- **`key_terms` unused in `measure`.** It is passed in but only `achievable` is used.
- **Pieces counted twice.** In normal mode `count_pieces(train)` runs twice (once for training, once for `piece_counts`). Harmless, slightly wasteful.
- **The rule depends on validation.** Because validation is held-out *patterns* (Day 5), this is a fair test of how the tokenizer handles unseen wording.
- **Day 5 token bounds.** The real token count of validation at the chosen size should fall between Day 5's lower bound and its UTF-8 byte count.

## 12. Quick summary in one paragraph

The script trains the byte-level BPE once on `train.txt`, cuts the result into many candidate vocabulary sizes, measures each on held-out validation text (compression, how many important glossary terms stay within 1–3 tokens, longest question), chooses the smallest size that satisfies a written rule, saves it, reloads it to prove nothing is lost, shows demos, and records everything (including SHA-256 fingerprints) in a report and the run log.
