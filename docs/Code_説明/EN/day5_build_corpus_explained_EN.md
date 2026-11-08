# day5_build_corpus.py — Simple Line-by-Line Explanation (English)

## 0. The big picture (read this first)

Imagine a teacher who has a big pile of **practice questions** and must prepare for a school exam.

- Most questions go to the **homework pile** (train): the student practises on them.
- Some go to a **quiz pile** (val = validation): the teacher uses it during the year to check progress.
- Some go to the **final exam pile** (test): the student must never see these until the very end.

The dangerous mistake is **cheating by accident**: if the exam contains the *same sentence with a tiny change* that the student already practised, the student looks smart but only memorised. This script is built to prevent that.

What it does, in one list:

| Step | Function | In plain words |
|---|---|---|
| 1 | `load_questions` | Read the questions and reject anything broken |
| 2 | `choose_held_out_patterns` | Pick **whole sentence patterns** (not single lines) for val and test |
| 3 | `build_splits` | Build the three piles, remove near-copies, cap big groups, repeat small types in train |
| 4 | `split_stats`, `coverage` | Count lines, characters, bytes, token bounds, vocabulary coverage |
| 5 | `write_lines`, `write_labels` | Save the piles as files |
| 6 | `write_data_card`, manifest JSON, `append_run_log` | Write the documentation and the "receipt" with fingerprints |
| 7 | sanity checks in `main` | Make sure nothing leaked between piles |

Run it like this:

```
.venv\Scripts\python.exe scripts\day5_build_corpus.py --machine home
```

The result is **fully determined** by the input files and `--seed`. So the school PC and home laptop produce byte-identical `train.txt`, `val.txt`, `test.txt` (you can prove it with the SHA-256 values in the manifest).

---

## 1. Words you need to know (mini dictionary)

| Word | Simple meaning |
|---|---|
| **train / val / test** | Practice pile / progress-check pile / final exam pile. |
| **Data leakage** | Test information sneaking into training. Makes scores look better than they really are. |
| **Pattern (文型)** | One sentence template (one *phrasing*). All questions from one pattern look almost the same. |
| **Question type** | A group of patterns that ask the same kind of thing (e.g. "ask for salary"). |
| **Held out** | Kept aside, away from training. |
| **Near-duplicate** | Two sentences that differ only in punctuation or spaces. |
| **NFKC normalisation** | A Unicode clean-up: converts full-width letters/digits (`ＡＢＣ１２３`) and similar variants to a standard form (`ABC123`). |
| **Oversampling** | Repeating items of a small group so the model sees it more often. |
| **Seed** | Starting number of a random generator. Same seed = same "random" results. |
| **Manifest** | A "receipt" file listing exactly what was made, with settings and fingerprints. |
| **Data card** | A document describing a dataset: sources, how it was made, size, limits. |
| **SHA-256** | A file fingerprint (see Day 4). |
| **Tokenizer / token** | The tool that cuts text into pieces ("tokens") for the model. A token is one piece. |
| **BPE (byte-level)** | A popular tokenizer method that starts from bytes and merges frequent pairs. |
| **Pre-tokenizer** | A first cutting step *before* BPE. BPE can never merge across the cuts it makes. |
| **Frozen dataclass** | A data-holding class whose objects cannot be changed after creation. |
| **Regular expression (regex)** | A mini-language to describe text patterns (here: "any of these punctuation characters"). |

---

## 2. The docstring (top of the file)

```python
r"""Day 5 - corpus split, light balancing, statistics and data card (Model A).
...
"""
```
- Same style as Day 1 and Day 4: raw (`r`) documentation string, English + Japanese, not executed.
- It lists the inputs (the team's corpus ZIP, `patterns.csv`, optional glossary and synonym files) and the 6 steps.
- Important sentence: **only the 質問 text is used for training; the 意味 labels go to separate `*_labels.csv` files.** (So the model never sees the answer key.)
- School-PC note: CPU + standard library only, so both machines give identical outputs.

---

## 3. Imports

```python
from __future__ import annotations
```
- Same as Day 1 and Day 4.

```python
import argparse
import csv
import hashlib
import io
import json
import math
import random
import re
import sys
import unicodedata
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable
```

`argparse`, `csv`, `hashlib`, `io`, `json`, `sys`, `zipfile`, `Counter`, `defaultdict`, `dataclass`, `datetime`, `Path` — **same as Day 4**. New ones:

| Line | What it gives you |
|---|---|
| `import math` | Math helpers. Here `math.ceil` (round **up**). |
| `import random` | Random numbers (shuffle, choice, sample). |
| `import re` | Regular expressions. |
| `import unicodedata` | Unicode tools, used for NFKC normalisation. |
| `from dataclasses import ... field` | `field(...)` lets a dataclass attribute have special settings (here: its own empty dict). |
| `from typing import Iterable` | A type meaning "anything you can loop over" (list, generator, ...). |

---

## 4. Constants

```python
SCRIPT_VERSION: str = "day5-v1"
REPO_ROOT: Path = Path(__file__).resolve().parent.parent
SPLITS: tuple[str, str, str] = ("train", "val", "test")
QUESTION_HEADER: list[str] = ["質問", "パターンID", "質問タイプID", "意味"]
```
- `SCRIPT_VERSION` and `REPO_ROOT` — same as Day 4.
- `SPLITS` — the three pile names in a fixed order, as a tuple of exactly 3 strings. Used in loops all over the script.
- `QUESTION_HEADER` — the exact header `questions.csv` must have (same as Day 4).

---

## 5. Data classes (the "boxes" that hold data)

### 5.1 `Question`

```python
@dataclass(frozen=True)
class Question:
    """One generated question and its internal labels."""

    text: str
    pattern_id: str
    type_id: str
    meaning: str  # JSON string, kept aside, never used as training text
```
- `@dataclass` (see Day 4) writes the constructor for you.
- `frozen=True` — after creating a `Question`, you **cannot change it**. This protects data from accidental edits, and as a bonus frozen objects can be put in sets or used as dictionary keys.
- Four fields: the sentence, which pattern made it, which question type it belongs to, and the JSON label (`meaning`). The comment reminds us the label is **never** training text.

### 5.2 `PatternInfo`

```python
@dataclass(frozen=True)
class PatternInfo:
    """One row of patterns.csv."""

    pattern_id: str
    type_id: str
    type_name: str
    style: str
    template: str
```
- One row of `patterns.csv` as an unchangeable box: pattern ID, type ID, type name, writing style (文体), and the sentence template (文型).

### 5.3 `SplitConfig` — all the settings in one place

```python
@dataclass
class SplitConfig:
    """All knobs of the split. Every value is written into the manifest."""

    seed: int = 42
    val_fraction: float = 0.07
    test_fraction: float = 0.07
    min_type_lines: int = 100
    min_pattern_lines: int = 20
    max_donate_share: float = 0.4
    eval_cap_per_pattern: int = 500
    oversample_target: int = 200
    max_repeat: int = 5
```
Each line is "name: type = default value". What each knob means:

| Setting | Meaning |
|---|---|
| `seed = 42` | Random starting number (repeatable results). |
| `val_fraction = 0.07` | About 7% of the (non-empty) patterns are held out for validation. |
| `test_fraction = 0.07` | About 7% held out for test. |
| `min_type_lines = 100` | A question type must have at least 100 questions to be allowed to give away a pattern. |
| `min_pattern_lines = 20` | A pattern given away must have at least 20 questions (not too tiny to measure anything). |
| `max_donate_share = 0.4` | ...and at most 40% of its type's questions (so train keeps most of every type). |
| `eval_cap_per_pattern = 500` | Keep at most 500 questions from each held-out pattern. |
| `oversample_target = 200` | Small types in train are repeated until they reach about 200 lines... |
| `max_repeat = 5` | ...but never more than 5 copies. |

Not frozen, so values can be replaced (command-line options do that). The docstring says every value is saved in the manifest, so you can always see which settings made a split.

### 5.4 `SplitResult`

```python
@dataclass
class SplitResult:
    assignment: dict[str, str]
    lines: dict[str, list[Question]]
    train_unique: list[Question]
    repeat_factor: dict[str, int]
    dropped_near_dup: dict[str, int] = field(default_factory=dict)
    dropped_by_cap: dict[str, int] = field(default_factory=dict)
```
The "bundle" returned by the splitting step:
- `assignment` — pattern ID → which pile (`"train"`, `"val"`, `"test"`).
- `lines` — pile name → its list of questions (train is already oversampled).
- `train_unique` — train **before** repeating (the real number of different questions).
- `repeat_factor` — type ID → how many copies in train.
- `dropped_near_dup`, `dropped_by_cap` — counts of removed lines per pile.
- `field(default_factory=dict)` — "if no value is given, make a **new empty dict for this object**." (Writing `= {}` directly is dangerous in Python because all objects would share the same dict.) The last two have defaults, so they come after the fields without defaults.

---

## 6. Loading

### 6.1 `sha256_of`

```python
def sha256_of(path: Path) -> str:
    """SHA-256 of a file, so anyone can prove the inputs and outputs match."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
```
Same goal as Day 4's `sha256_file`, but done **in pieces** so a huge file does not need to fit in memory.
- `h = hashlib.sha256()` — create an empty hash calculator.
- `path.open("rb")` — open for reading raw bytes (`b` = binary).
- `iter(lambda: f.read(1 << 20), b"")` — keep calling `f.read(1 << 20)` (read 1 MiB = 1,048,576 bytes) until it returns an empty `b""` (end of file). `lambda` is a tiny one-line function; `1 << 20` means "1 shifted left 20 bits" = 2²⁰.
- `h.update(chunk)` — feed each piece into the calculator.
- `h.hexdigest()` — the final fingerprint as text.

### 6.2 `read_questions_text`

```python
def read_questions_text(source: Path) -> str:
    """Return the text of questions.csv, read from a .zip or a plain .csv."""
    if source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as zf:
            members = [n for n in zf.namelist() if n.endswith("questions.csv")]
            if len(members) != 1:
                raise SystemExit(f"{source}: expected exactly one questions.csv inside, found {members}")
            return zf.read(members[0]).decode("utf-8-sig")
    return source.read_text(encoding="utf-8-sig")
```
- Same logic as Day 4's `read_corpus_bytes` (ZIP → find exactly one `questions.csv`; otherwise plain file). The only difference: it returns **text** (decoded with `utf-8-sig`, which removes the BOM) instead of raw bytes, because here we do not need byte-level comparison.

### 6.3 `load_questions`

```python
def load_questions(source: Path) -> list[Question]:
    """Load and validate the question corpus. Stops with a clear message on bad data."""
    reader = csv.reader(io.StringIO(read_questions_text(source)))
    header = next(reader)
    if header != QUESTION_HEADER:
        raise SystemExit(f"Unexpected header {header}; expected {QUESTION_HEADER}")
```
- Read the text, wrap it as a file-like object, create a CSV reader, take the first row as the header and check it (same idea as Day 4's `parse_corpus`).

```python
    questions: list[Question] = []
    seen: set[str] = set()
    type_of_pattern: dict[str, str] = {}
    problems: list[str] = []
```
- Four containers:
  - `questions` — the good questions we accept.
  - `seen` — a set of sentences already accepted (to find duplicates quickly).
  - `type_of_pattern` — pattern ID → its type ID (to make sure one pattern never has two types).
  - `problems` — descriptions of problems found.

```python
    for line_no, row in enumerate(reader, start=2):
```
- `enumerate(reader, start=2)` gives a counter **and** the row. It starts at 2 because line 1 was the header. So messages can say "line 57" matching the real file line.

```python
        if len(row) != 4:
            problems.append(f"line {line_no}: {len(row)} fields")
            continue
```
- **Check A:** exactly 4 columns. If not, record a problem and `continue` (skip to the next row). *This is more careful than Day 4:* here a bad row is skipped safely, so the script never crashes while unpacking it.

```python
        text, pid, tid, meaning = (c.strip() for c in row)
```
- Unpack the 4 columns into 4 names, with `.strip()` removing spaces at both ends of each cell (a generator expression feeding the unpacking).

```python
        if not text or "\n" in text or "\r" in text:
            problems.append(f"line {line_no}: empty or multi-line question")
            continue
```
- **Check B:** the question must not be empty and must not contain a line break (`\n` or `\r`), because the training file stores **one question per line**.

```python
        if text in seen:
            problems.append(f"line {line_no}: duplicate question {text!r}")
            continue
```
- **Check C:** duplicates. `{text!r}` shows the text with quotes (its `repr`), which makes invisible spaces easier to spot.

```python
        if type_of_pattern.setdefault(pid, tid) != tid:
            problems.append(f"line {line_no}: pattern {pid} has two types")
            continue
```
- **Check D:** one pattern, one type. `setdefault(pid, tid)` stores `tid` if this pattern is new and returns whatever is stored. If the stored type differs from this row's type, the pattern is inconsistent.

```python
        try:
            label = json.loads(meaning)
        except json.JSONDecodeError:
            problems.append(f"line {line_no}: 意味 is not valid JSON")
            continue
        if label.get("パターン") != pid or label.get("質問タイプ") != tid:
            problems.append(f"line {line_no}: 意味 does not match its pattern/type columns")
            continue
```
- **Check E:** the label must be valid JSON, and the pattern/type stored inside it must match the row's columns (same check as Day 4).

```python
        seen.add(text)
        questions.append(Question(text, pid, tid, meaning))
```
- Only if the row passed **all** checks: remember its sentence and add a `Question`.

```python
    if problems:
        print(f"{len(problems)} problem(s) in the corpus; first 10:")
        for p in problems[:10]:
            print("  ", p)
        raise SystemExit("Fix the corpus (or regenerate it with the team's generator) and run again.")
    return questions
```
- If there are any problems: print how many, show the first 10 (`problems[:10]` = slice of the first ten), then stop. Otherwise return the clean list.

### 6.4 `load_patterns`

```python
def load_patterns(path: Path) -> dict[str, PatternInfo]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    return {
        r["パターンID"]: PatternInfo(r["パターンID"], r["質問タイプID"], r["質問タイプ"], r["文体"], r["文型"])
        for r in rows
    }
```
- Read `patterns.csv` as dictionaries (same as Day 4's `load_pattern_table`), then build a **dictionary comprehension**: pattern ID → `PatternInfo` box. This allows quick lookups like `patterns["P012"].type_id`.

### 6.5 `load_terms`

```python
def load_terms(glossary: Path, synonyms: Path) -> tuple[list[str], list[str]]:
    """Standard terms and their variants, for the coverage report. Missing files -> empty lists."""
    terms: list[str] = []
    variants: list[str] = []
    if glossary.exists():
        with glossary.open(encoding="utf-8-sig", newline="") as f:
            terms = sorted({r["標準用語"].strip() for r in csv.DictReader(f) if r["標準用語"].strip()})
    if synonyms.exists():
        with synonyms.open(encoding="utf-8-sig", newline="") as f:
            variants = sorted({r["変形・同義語"].strip() for r in csv.DictReader(f) if r["変形・同義語"].strip()})
    return terms, variants
```
- Start with two empty lists. These files are **optional**, so a missing file simply stays empty.
- `{ ... for r in ... if ...}` is a **set comprehension**: take the `標準用語` (standard term) column, strip spaces, skip empty cells, and remove duplicates automatically (a set). `sorted(...)` makes it a list in a fixed order.
- The second block (variants / `変形・同義語` column) is the same idea for the synonym dictionary.

---

## 7. Splitting

### 7.1 Near-duplicate key

```python
_NEAR_DUP_STRIP = re.compile(r"[\s。、，．,.!！?？・「」『』（）()〜~ー－\-]")
```
- `re.compile(...)` builds a regular expression once and reuses it. A leading `_` means "private to this file".
- Inside `[ ... ]` is a **character set**: match any one of: whitespace (`\s`), Japanese and Western punctuation, brackets, wave dash, long-vowel mark `ー`, hyphens.

```python
def near_dup_key(text: str) -> str:
    """Text with punctuation, spaces and long-vowel marks removed (after NFKC)."""
    return _NEAR_DUP_STRIP.sub("", unicodedata.normalize("NFKC", text))
```
- First `unicodedata.normalize("NFKC", text)` makes variants uniform (full-width → normal).
- Then `.sub("", ...)` replaces every matched character with nothing (deletes it).
- Result = the **fingerprint of the wording**. Example: `給与は？` and `給与は` give the same key, so they count as near-duplicates.

### 7.2 `choose_held_out_patterns` — which patterns go to val/test

```python
def choose_held_out_patterns(
    questions: list[Question], patterns: dict[str, PatternInfo], cfg: SplitConfig
) -> dict[str, str]:
```
- Input: all questions, the pattern table, the settings. Output: a dictionary pattern ID → pile name for **every** pattern that has questions.
- The docstring explains the rules (all deterministic for a given seed):
  - Only types with ≥ `min_type_lines` questions and ≥ 3 non-empty patterns may give away a pattern → every type keeps **at least 2 phrasings in train**.
  - Each chosen type gives **exactly one** pattern; val types and test types are **different** types → held-out sets cover many types.
  - The pattern given must have ≥ `min_pattern_lines` questions and ≤ `max_donate_share` of its type.

```python
    rng = random.Random(cfg.seed)
```
- Creates **our own** random generator with the seed. It does not touch Python's global random state, so the result depends only on the seed.

```python
    per_pattern = Counter(q.pattern_id for q in questions)
    per_type = Counter(q.type_id for q in questions)
```
- Count questions per pattern and per type. (Two lines, same idea, different key.)

```python
    patterns_of_type: dict[str, list[str]] = defaultdict(list)
    for pid in sorted(per_pattern):
        patterns_of_type[patterns[pid].type_id].append(pid)
```
- Build "type → list of its non-empty patterns". Looping over `sorted(...)` keeps the order fixed, which keeps the random choices repeatable. `defaultdict(list)` starts missing keys as empty lists.

```python
    def candidates_of(tid: str) -> list[str]:
        return [
            p for p in patterns_of_type[tid]
            if cfg.min_pattern_lines <= per_pattern[p] <= cfg.max_donate_share * per_type[tid]
        ]
```
- A helper function: which patterns of type `tid` are **allowed** to be given away? The chained comparison `a <= x <= b` means "x is between a and b": at least 20 questions and at most 40% of the type's total.

```python
    eligible: list[str] = []
    for tid in sorted(patterns_of_type):
        candidates = candidates_of(tid)
        if per_type[tid] >= cfg.min_type_lines and len(patterns_of_type[tid]) >= 3 and candidates:
            eligible.append(tid)
```
- Collect the **eligible types**: enough questions (≥100), at least 3 patterns, and at least one allowed candidate pattern. (`and candidates` is true only if the list is not empty.)

```python
    n_patterns = len(per_pattern)
    n_val = round(cfg.val_fraction * n_patterns)
    n_test = round(cfg.test_fraction * n_patterns)
```
- How many patterns to hold out: 7% of all non-empty patterns, rounded, for val; same for test. (Second line = same idea with the test fraction.)

```python
    if n_val + n_test > len(eligible):
        print(f"WARNING: only {len(eligible)} eligible types; reducing held-out patterns.")
        n_val = len(eligible) // 2
        n_test = len(eligible) - n_val
```
- Safety net: if we want more held-out patterns than there are eligible types (each type gives only one), print a warning and split the eligible types in half (`//` = whole-number division).

```python
    rng.shuffle(eligible)
    assignment = {pid: "train" for pid in per_pattern}
    for split, chosen_types in (("val", eligible[:n_val]), ("test", eligible[n_val:n_val + n_test])):
        for tid in sorted(chosen_types):
            assignment[rng.choice(candidates_of(tid))] = split
    return assignment
```
- `rng.shuffle(eligible)` — shuffle the eligible types in place (seeded, repeatable).
- Start by putting **every** pattern in `"train"`.
- Take the first `n_val` types for validation, the next `n_test` types for test (two slices that do not overlap → the type sets are disjoint).
- For each chosen type (in sorted order, again for repeatability): pick one random allowed pattern with `rng.choice(...)` and change its pile to `val` or `test`.
- Return the assignment.

### 7.3 `build_splits` — build the three piles

```python
def build_splits(questions: list[Question], assignment: dict[str, str], cfg: SplitConfig) -> SplitResult:
    """Apply the pattern assignment, remove near-duplicates, cap eval patterns, oversample train."""
    rng = random.Random(cfg.seed + 1)
```
- A new seeded generator (`seed + 1`, so it is a different stream than the one in the previous function, but still repeatable).

```python
    by_split: dict[str, list[Question]] = {s: [] for s in SPLITS}
    for q in questions:
        by_split[assignment[q.pattern_id]].append(q)
```
- Make three empty lists, then put every question into the pile its **pattern** was assigned to. (Whole patterns move together; this is the "split by pattern" idea.)

**Step A: near-duplicate guard**

```python
    dropped_near_dup: dict[str, int] = {}
    blocked = {near_dup_key(q.text) for q in by_split["train"]}
    for split in ("val", "test"):
        kept = [q for q in by_split[split] if near_dup_key(q.text) not in blocked]
        dropped_near_dup[split] = len(by_split[split]) - len(kept)
        by_split[split] = kept
        blocked |= {near_dup_key(q.text) for q in kept}
```
- `blocked` = the wording-fingerprints of every train question ("already seen by the student").
- For val, then for test: keep only questions whose fingerprint is **not** blocked; count how many were dropped; replace the pile with the kept ones.
- `blocked |= {...}` — add the kept questions' fingerprints to `blocked` too. Result: **train wins; val loses to train; test loses to train and val.** No pile shares a wording with another.

**Step B: cap held-out patterns**

```python
    dropped_by_cap: dict[str, int] = {}
    for split in ("val", "test"):
        grouped: dict[str, list[Question]] = defaultdict(list)
        for q in by_split[split]:
            grouped[q.pattern_id].append(q)
        capped: list[Question] = []
        dropped = 0
        for pid in sorted(grouped):
            items = grouped[pid]
            if len(items) > cfg.eval_cap_per_pattern:
                dropped += len(items) - cfg.eval_cap_per_pattern
                items = rng.sample(items, cfg.eval_cap_per_pattern)
            capped.extend(items)
        dropped_by_cap[split] = dropped
        by_split[split] = capped
```
- For val and test: group questions by pattern ID.
- For each pattern (sorted order): if it has more than 500 questions, count the extras as dropped and keep a **random sample** of exactly 500 (`rng.sample` picks without repeats). Add the kept ones to `capped`.
- Save the dropped count and replace the pile. The comment in the code says the dropped lines are **not** moved to train — they are simply thrown away, otherwise the held-out pattern would leak into training.
- Why cap? One giant pattern could make up most of the exam and distort the score.

**Step C: oversample small types in train only**

```python
    train_unique = list(by_split["train"])
    per_type = Counter(q.type_id for q in train_unique)
    repeat_factor = {
        tid: min(cfg.max_repeat, max(1, math.ceil(cfg.oversample_target / n))) for tid, n in per_type.items()
    }
```
- `train_unique` — a **copy** of train before repeating (`list(...)` makes a copy).
- Count train questions per type.
- The formula for how many copies each type gets:
  `ceil(200 / n)` rounded up → at least 1 (`max(1, ...)`) → at most 5 (`min(5, ...)`).
  Examples: a type with 40 questions → ceil(5.0)=5 copies; 150 → ceil(1.33)=2; 500 → ceil(0.4)=1 → stays 1; 10 → ceil(20)=20 → capped to 5.

```python
    train: list[Question] = []
    for q in train_unique:
        train.extend([q] * repeat_factor[q.type_id])
    by_split["train"] = train
```
- For each train question, add `repeat_factor` copies. `[q] * 3` makes a list with 3 copies of the same object; `extend` adds them all. Replace the train pile with the bigger list.

```python
    for split in SPLITS:
        rng.shuffle(by_split[split])
    return SplitResult(assignment, by_split, train_unique, repeat_factor, dropped_near_dup, dropped_by_cap)
```
- Shuffle each pile (seeded), so the lines are not grouped by pattern (and the repeated copies get mixed in).
- Return everything in a `SplitResult`.

---

## 8. Statistics

### 8.1 `script_class`

```python
def script_class(ch: str) -> str:
    """Script class of one character, matching the planned Day 6 pre-tokenizer."""
    o = ord(ch)
```
- Tells which **kind of character** `ch` is. `ord(ch)` = its Unicode number (code point). The planned Day 6 pre-tokenizer will cut text using the same classes.

```python
    if 0x3041 <= o <= 0x309F:
        return "hiragana"
```
- Numbers between 0x3041 and 0x309F (hexadecimal) are **hiragana**.

```python
    if (0x30A1 <= o <= 0x30FA) or o in (0x30FC, 0x30FD, 0x30FE) or 0x31F0 <= o <= 0x31FF or 0xFF66 <= o <= 0xFF9F:
        return "katakana"
```
- **Katakana**: the main block, plus the long-vowel mark `ー` and the repeat marks (0x30FC–0x30FE), small katakana extensions, and half-width katakana.

```python
    if 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF or o in (0x3005, 0x3006, 0x30F6):
        return "kanji"
```
- **Kanji**: the main CJK block, the extension A block, plus special characters `々`, `〆`, `ヶ` that behave like kanji.

```python
    if ch.isascii() and ch.isalpha() or 0xFF21 <= o <= 0xFF3A or 0xFF41 <= o <= 0xFF5A:
        return "latin"
```
- **Latin letters**: normal ASCII letters, or full-width A–Z / a–z. (Python evaluates `and` before `or`, so this reads "(ASCII and a letter) or full-width upper or full-width lower".)

```python
    if ch.isdigit():
        return "digit"
    if ch.isspace():
        return "space"
    return "other"
```
- Digits (this includes full-width digits), spaces, and everything else (punctuation, symbols) → `"other"`.

### 8.2 `pretoken_chunks`

```python
def pretoken_chunks(text: str) -> int:
    chunks = 0
    previous = ""
    for ch in text:
        cls = script_class(ch)
        if cls != previous or cls == "other":
            chunks += 1
        previous = cls
    return chunks
```
- Counts **runs of the same character class**. A new chunk starts when the class changes, **or** when the character is `"other"` (so every punctuation mark is its own chunk).
- Example: `給与はいくら？` → kanji `給与` (1) → hiragana `はいくら` (2) → other `？` (3) → **3 chunks**.
- Why it matters (from the docstring): a byte-level BPE using this pre-tokenizer **cannot merge across chunks**, so every chunk is at least one token. Therefore the chunk count is a hard **lower bound** on the token count. The number of UTF-8 bytes is the **upper bound** (at worst one token per byte). The real count will be measured on Day 7 with the team's tokenizer.

### 8.3 `split_stats`

```python
def split_stats(qs: list[Question]) -> dict[str, int]:
    texts = [q.text for q in qs]
    return {
        "lines": len(texts),
        "unique_lines": len(set(texts)),
        "characters": sum(len(t) for t in texts),
        "utf8_bytes": sum(len(t.encode("utf-8")) for t in texts),
        "token_lower_bound": sum(pretoken_chunks(t) for t in texts),
        "patterns": len({q.pattern_id for q in qs}),
        "types": len({q.type_id for q in qs}),
        "max_chars": max((len(t) for t in texts), default=0),
    }
```
- Collect the sentences, then compute:
  - `lines` — how many; `unique_lines` — how many different (set removes repeats; for train this shows the effect of oversampling).
  - `characters` — total characters. `utf8_bytes` — total bytes (`t.encode("utf-8")` converts text to bytes; Japanese characters take 3 bytes each).
  - `token_lower_bound` — sum of chunks over all sentences.
  - `patterns`, `types` — how many different ones appear.
  - `max_chars` — the longest line. `default=0` avoids an error if the pile is empty.

### 8.4 `coverage`

```python
def coverage(texts: Iterable[str], words: list[str]) -> int:
    """How many of `words` occur at least once in the texts (substring match)."""
    blob = "\n".join(texts)
    return sum(1 for w in words if w in blob)
```
- Glue all sentences into one big string (`blob`).
- For each word, `w in blob` is true if the word appears anywhere inside it. `sum(1 for ...)` counts the true ones.
- It is a **substring** check, so a short word can match inside a longer word (the data card warns about this).

---

## 9. Writing files

### 9.1 `write_lines`

```python
def write_lines(path: Path, qs: list[Question]) -> None:
    """One question per line, UTF-8 without BOM, LF line endings (what the tokenizer reads)."""
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for q in qs:
            f.write(q.text + "\n")
```
- `"w"` = write (overwrites). `encoding="utf-8"` = UTF-8 **without** BOM.
- `newline="\n"` — always use plain LF line endings, even on Windows (which normally writes CRLF). This keeps the file identical on both machines (same SHA-256).
- Write each question followed by a line break. Only the question text — **no labels**.

### 9.2 `write_labels`

```python
def write_labels(path: Path, qs: list[Question], split: str) -> None:
    """Internal label file (not training text). Same order as the .txt file."""
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["質問", "パターンID", "質問タイプID", "split", "意味"])
        for q in qs:
            w.writerow([q.text, q.pattern_id, q.type_id, split, q.meaning])
```
- A CSV with the labels, in the **same order** as the `.txt`, so line N of the text matches row N of the labels.
- `utf-8-sig` writes a BOM so Excel opens the Japanese correctly. `newline=""` is what the `csv` module needs.
- `csv.writer` handles quotes and commas inside cells automatically (important because `meaning` is JSON, full of commas and quotes).
- First write a header row, then one row per question, including the `split` name.

### 9.3 `fmt`

```python
def fmt(n: int | float) -> str:
    return f"{n:,}" if isinstance(n, int) else f"{n:,.2f}"
```
- Formats numbers for the report: whole numbers with thousands commas (`12,345`), decimals with commas and 2 decimal places (`1,234.50`).

### 9.4 `write_data_card`

```python
def write_data_card(path, stats, per_type_rows, result, patterns, empty_patterns, cov, n_terms, n_variants, cfg, source_name, source_sha) -> None:
    """docs/DATA_CARD.md. Every number comes from this run; nothing is typed by hand."""
```
- Writes the data card (a Markdown document). The parameters are everything it needs: statistics, the per-type table rows, the split result, pattern table, empty patterns, coverage, term counts, settings, input file name and fingerprint.

```python
    held = {s: sorted(p for p in ... if sp == s) for s in ("val", "test")}
    style_counts = {s: Counter(patterns[p].style for p in held[s]) for s in held}
```
- `held` — for val and test, the sorted list of pattern IDs in that pile. (Written as one dict comprehension.)
- `style_counts` — for each of those piles, how many held-out patterns have each writing style.

```python
    lines: list[str] = [ ... ]
```
- Then one big list of Markdown lines. It is documentation text, so here is a map of the sections instead of every sentence:

| Section | What it says | Where its numbers come from |
|---|---|---|
| Title + date | Who generated it and when | `SCRIPT_VERSION`, `datetime.now()` |
| 1. What the corpus is | Short Japanese questions about company employee data | fixed text |
| 2. Sources and license | A table of what the team made; "no outside text is used"; the input file name and its SHA-256 | `source_name`, `source_sha` |
| 3. How it was split | By pattern, not by line; how many patterns are held out; rules; near-duplicates removed; cap; seed; held-out styles | `held`, `cfg`, `result.dropped_*`, `style_counts` |
| 4. Balancing | Oversampling only in train | `cfg.oversample_target`, `cfg.max_repeat` |
| 5. Size | A table per split | `stats` |
| 6. Per question type | A table per type | `per_type_rows` |
| 7. Vocabulary coverage | How many glossary terms / variants appear | `cov`, `n_terms`, `n_variants` |
| 8. Known limits and biases | Honest list of weaknesses (narrow domain, template-made, repeated names, uneven types, empty patterns, ambiguous words) | fixed text + `empty_patterns` |
| 9. Reproduce | The command to run again | fixed text |

Notable formatting tricks:
- `{cfg.max_donate_share:.0%}` — shows 0.4 as `40%`.
- `{', '.join(empty_patterns) or 'none'}` — joins the IDs, or shows `none` if the list is empty (same trick as Day 4).
- `{dict(style_counts['val'])}` — turns a `Counter` into a plain dict for printing.
- `".venv\\Scripts\\python.exe ..."` — inside a normal (non-raw) string a backslash must be doubled (`\\`) to appear as one `\`.
- Some lines start with `f"` but have no `{}`; that is harmless.

```python
    for s in SPLITS:
        st = stats[s]
        lines.append(
            f"| {s} | {fmt(st['lines'])} | ... |"
        )
```
- Adds one table row per pile to the **Size** table, using `fmt` for number formatting.

```python
    lines += [ ... ]
```
- Continues the list with the rest of the text: the explanation of token bounds, the per-type table (`*("| " + " | ".join(r) + " |" for r in per_type_rows)` — the `*` unpacks one Markdown row per type, built by joining the row's strings with ` | `), the coverage table (`*(f"| {s} | ..." for s in SPLITS)`), the limits, and the reproduce command.

```python
    path.write_text("\n".join(lines), encoding="utf-8")
```
- Join all lines with line breaks and save the file.

### 9.5 `append_run_log`

```python
def append_run_log(log, machine, stats, manifest_rel, cfg, n_val, n_test) -> None:
    entry = [
        "",
        f"## {datetime.now():%Y-%m-%d %H:%M} | {machine} | Day 5 corpus split",
        "",
        f"- Machine label: {machine} (CPU only; no GPU used)",
        f"- Script: scripts/day5_build_corpus.py ({SCRIPT_VERSION}), seed {cfg.seed}",
        f"- Held-out patterns: validation {n_val}, test {n_test} (split by pattern, not by line)",
    ]
    for s in SPLITS:
        st = stats[s]
        entry.append(
            f"- {s}: {st['lines']:,} lines ({st['unique_lines']:,} unique), {st['characters']:,} chars, "
            f"token bounds {st['token_lower_bound']:,} to {st['utf8_bytes']:,}"
        )
    entry += [f"- Manifest: {manifest_rel}", "- Result: PASS", ""]
    with log.open("a", encoding="utf-8") as f:
        f.write("\n".join(entry))
```
- Same pattern as Day 1 and Day 4: build a list of Markdown lines, then append to `run_log.md`.
- The loop adds one line per pile with its numbers and token bounds (lower bound `to` upper bound).
- `"- Result: PASS"` is always written, because any failure raises `SystemExit` before this point.

---

## 10. Main part

### 10.1 `parse_args`

```python
def parse_args() -> argparse.Namespace:
    d = SplitConfig()
    ap = argparse.ArgumentParser(description="Day 5: split the question corpus by pattern and write the data card.")
```
- `d = SplitConfig()` — creates a settings object just to read its **default values**. The defaults are written once (in `SplitConfig`) and reused for the command-line defaults. No duplication.

```python
    ap.add_argument("--machine", choices=["school", "home"], required=True, help="label for run_log.md")
    ap.add_argument("--source", type=Path, default=REPO_ROOT / "data/source/questions_corpus.zip", help="questions_corpus.zip or questions.csv")
    ap.add_argument("--patterns", type=Path, default=REPO_ROOT / "data/source/patterns.csv")
    ap.add_argument("--glossary", type=Path, default=REPO_ROOT / "data/source/用語集.csv")
    ap.add_argument("--synonyms", type=Path, default=REPO_ROOT / "data/source/同義語辞書.csv")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data/corpus")
    ap.add_argument("--manifest", type=Path, default=REPO_ROOT / "data/splits/split_manifest.json")
    ap.add_argument("--data-card", type=Path, default=REPO_ROOT / "docs/DATA_CARD.md")
    ap.add_argument("--run-log", type=Path, default=REPO_ROOT / "run_log.md")
    ap.add_argument("--no-log", action="store_true", help="do not append to run_log.md (for dry runs)")
```
- Same `argparse` idea as Day 1 and Day 4. `--machine` is required.
- The `Path` options (`--source`, `--patterns`, `--glossary`, `--synonyms`, `--out-dir`, `--manifest`, `--data-card`, `--run-log`) all work the same way: a path with a default location. Only the name and default differ.
- `--no-log` — a flag: if written, do not append to `run_log.md`. The help text says it is for **dry runs**.

```python
    ap.add_argument("--seed", type=int, default=d.seed)
    ap.add_argument("--val-fraction", type=float, default=d.val_fraction)
    ap.add_argument("--test-fraction", type=float, default=d.test_fraction)
    ap.add_argument("--min-type-lines", type=int, default=d.min_type_lines)
    ap.add_argument("--min-pattern-lines", type=int, default=d.min_pattern_lines)
    ap.add_argument("--max-donate-share", type=float, default=d.max_donate_share)
    ap.add_argument("--eval-cap", type=int, default=d.eval_cap_per_pattern)
    ap.add_argument("--oversample-target", type=int, default=d.oversample_target)
    ap.add_argument("--max-repeat", type=int, default=d.max_repeat)
    return ap.parse_args()
```
- One option for each knob in `SplitConfig`. All nine lines are the same idea: `type=int` or `type=float`, with the default taken from `d`.
- `ap.parse_args()` reads what you typed and returns it.

### 10.2 `main` — setup and loading

```python
def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
```
- Same as Day 4. The comment explains why: a Japanese error message must never crash a console that is not UTF-8 (such as `cp932` on Japanese Windows).

```python
    a = parse_args()
    for name in ("source", "patterns", "glossary", "synonyms", "out_dir", "manifest", "data_card", "run_log"):
        setattr(a, name, getattr(a, name).resolve())
```
- Same `setattr`/`getattr` loop as Day 4, now for 8 path options: make each an absolute path so relative paths typed on the command line also work.

```python
    cfg = SplitConfig(a.seed, a.val_fraction, a.test_fraction, a.min_type_lines, a.min_pattern_lines,
                      a.max_donate_share, a.eval_cap, a.oversample_target, a.max_repeat)
```
- Build the real settings object from the command-line values. The values are given **in the same order** as the fields in `SplitConfig`.

```python
    for p in (a.source, a.patterns):
        if not p.exists():
            raise SystemExit(f"Missing input: {p}")
```
- The corpus and `patterns.csv` are **required**; stop if either is missing. (The glossary and synonym files are optional.)

```python
    print(f"=== Day 5 corpus split | machine={a.machine} | seed={cfg.seed} ===")
    questions = load_questions(a.source)
    patterns = load_patterns(a.patterns)
```
- Print a header, load and validate the questions, load the pattern table.

```python
    unknown = sorted({q.pattern_id for q in questions} - patterns.keys())
    if unknown:
        raise SystemExit(f"Pattern IDs not in patterns.csv: {unknown[:10]}")
```
- Set subtraction: pattern IDs used in the questions **minus** the IDs known in `patterns.csv`. Anything left is an unknown ID → stop (showing up to 10).

```python
    wrong_type = sorted({q.pattern_id for q in questions if patterns[q.pattern_id].type_id != q.type_id})
    if wrong_type:
        raise SystemExit(f"Type in corpus differs from patterns.csv for: {wrong_type[:10]}")
```
- Cross-check: for each question, the type in the corpus must equal the type in `patterns.csv`. If not, stop.

```python
    used = {q.pattern_id for q in questions}
    empty_patterns = sorted(set(patterns) - used)
    print(f"Loaded {len(questions):,} questions, {len(used)} patterns with questions, "
          f"{len({q.type_id for q in questions})} types.")
    if empty_patterns:
        print(f"Patterns with no questions (all their sentences already came from another pattern): {empty_patterns}")
```
- `used` — patterns that have questions. `empty_patterns` — patterns that exist in `patterns.csv` but have no questions (all their sentences were duplicates of earlier patterns).
- Print a summary, and a note if there are empty patterns.

```python
    assignment = choose_held_out_patterns(questions, patterns, cfg)
    result = build_splits(questions, assignment, cfg)
```
- **The two key steps:** choose which patterns are held out, then build the piles.

### 10.3 `main` — writing outputs

```python
    a.out_dir.mkdir(parents=True, exist_ok=True)
    a.manifest.parent.mkdir(parents=True, exist_ok=True)
    a.data_card.parent.mkdir(parents=True, exist_ok=True)
```
- Make sure the three output folders exist (same `mkdir` idea as before; the manifest and data card need their parent folders).

```python
    outputs: dict[str, str] = {}
    for s in SPLITS:
        txt = a.out_dir / f"{s}.txt"
        lab = a.out_dir / f"{s}_labels.csv"
        write_lines(txt, result.lines[s])
        write_labels(lab, result.lines[s], s)
        outputs[str(txt.relative_to(REPO_ROOT)).replace("\\", "/")] = sha256_of(txt)
        outputs[str(lab.relative_to(REPO_ROOT)).replace("\\", "/")] = sha256_of(lab)
```
- For each pile, write `train.txt` / `train_labels.csv` (and the same for val, test).
- Record each file's fingerprint in `outputs`. The key is the path relative to the repo with `/` instead of `\`, so it looks the same on Windows and elsewhere (same trick as Day 4's `work_rel`).

```python
    stats = {s: split_stats(result.lines[s]) for s in SPLITS}
    stats["train_unique"] = split_stats(result.train_unique)
```
- Statistics per pile, plus an extra entry for train **before** repeating.

```python
    tr_u = Counter(q.type_id for q in result.train_unique)
    tr = Counter(q.type_id for q in result.lines["train"])
    va = Counter(q.type_id for q in result.lines["val"])
    te = Counter(q.type_id for q in result.lines["test"])
    type_names = {p.type_id: p.type_name for p in patterns.values()}
    per_type_rows = [
        [t, type_names.get(t, ""), str(tr_u[t]), str(result.repeat_factor.get(t, 1)), str(tr[t]), str(va[t]), str(te[t])]
        for t in sorted({q.type_id for q in questions})
    ]
```
- Four `Counter`s count questions per type in: train-unique, train (after repeats), val, test. (Four lines, same idea.) A `Counter` returns 0 for a missing key, so `va[t]` is safe even if that type has no val questions.
- `type_names` — type ID → readable name.
- `per_type_rows` — a list of rows for the data card. Each row is a list of **strings**: type ID, name, train unique, copies, train lines, val, test.

```python
    terms, variants = load_terms(a.glossary, a.synonyms)
    cov = {
        s: {"terms": coverage((q.text for q in result.lines[s]), terms),
            "variants": coverage((q.text for q in result.lines[s]), variants)}
        for s in SPLITS
    }
    if not terms:
        print("NOTE: glossary/synonym files not found; the coverage section will show 0.")
```
- Load the glossary and synonym lists.
- `cov` — for each pile, how many glossary terms and how many variants appear in its text. A nested dictionary comprehension; `(q.text for q in ...)` is a generator that feeds `coverage`.
- If no terms were loaded, print a note.

```python
    held = {s: sorted(p for p, sp in assignment.items() if sp == s) for s in SPLITS}
```
- For each pile (including train): the sorted list of its pattern IDs.

```python
    manifest = {
        "script": "scripts/day5_build_corpus.py",
        "script_version": SCRIPT_VERSION,
        "created": datetime.now().isoformat(timespec="seconds"),
        "machine": a.machine,
        "python": sys.version.split()[0],
        "config": cfg.__dict__,
        "inputs": {
            str(a.source.name): sha256_of(a.source),
            str(a.patterns.name): sha256_of(a.patterns),
        },
        "outputs_sha256": outputs,
        "patterns_per_split": held,
        "patterns_without_questions": empty_patterns,
        "repeat_factor_per_type": dict(sorted(result.repeat_factor.items())),
        "dropped_near_duplicates": result.dropped_near_dup,
        "dropped_by_eval_cap": result.dropped_by_cap,
        "stats": stats,
        "coverage": {"glossary_terms": len(terms), "synonym_variants": len(variants), **cov},
    }
```
- The **manifest**: the "receipt" of this run. Contents: which script and version, when (`isoformat(timespec="seconds")` → `2026-11-08T14:05:33`), which machine, Python version (`sys.version.split()[0]`, just the number), all settings (`cfg.__dict__` = the dataclass fields as a dictionary), fingerprints of the **inputs** and **outputs**, which patterns went where, repeat factors, what was dropped, statistics, and coverage.
- `**cov` unpacks the `cov` dictionary into the dictionary being built (merging the two).
- Note: `created`, `machine`, `python` differ between machines, but `outputs_sha256` should be identical. So compare `outputs_sha256`, not the whole file.

```python
    a.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    write_data_card(a.data_card, stats, per_type_rows, result, patterns, empty_patterns, cov,
                    len(terms), len(variants), cfg, a.source.name, manifest["inputs"][a.source.name])
```
- Save the manifest as readable JSON (same as Day 1 and Day 4). Then write the data card, passing the input file's fingerprint from the manifest.

### 10.4 `main` — safety checks, summary, log

```python
    train_keys = {near_dup_key(q.text) for q in result.train_unique}
    leak = sum(1 for s in ("val", "test") for q in result.lines[s] if near_dup_key(q.text) in train_keys)
    overlap = set(held["val"]) & set(held["test"]) | (set(held["val"]) | set(held["test"])) & set(held["train"])
    if leak or overlap:
        raise SystemExit(f"Split check FAILED: {leak} leaked lines, {len(overlap)} shared patterns.")
```
- **A final inspection of what was written.**
- `train_keys` — wording-fingerprints of train.
- `leak` — count val/test lines whose fingerprint appears in train. Should be 0.
- `overlap` — patterns that appear in more than one pile. Operator `&` (intersection) runs before `|` (union), so this means: **(val ∩ test) ∪ ((val ∪ test) ∩ train)**. Should be empty.
- If either is non-zero, stop with an error ("fail loudly if the split leaked").

```python
    print()
    print(f"{'split':<6} {'lines':>9} {'unique':>9} {'chars':>11} {'token bounds':>25} {'patterns':>9} {'types':>6}")
    for s in SPLITS:
        st = stats[s]
        bounds = f"{st['token_lower_bound']:,} - {st['utf8_bytes']:,}"
        print(f"{s:<6} {st['lines']:>9,} {st['unique_lines']:>9,} {st['characters']:>11,} {bounds:>25} "
              f"{st['patterns']:>9} {st['types']:>6}")
```
- Prints a neat console table. In f-strings, `<6` means left-align in width 6, `>9` means right-align in width 9, `,` adds thousands commas. `bounds` shows `lower - upper` token bounds.

```python
    print(f"Near-duplicates removed: {result.dropped_near_dup}; over eval cap: {result.dropped_by_cap}")
    oversampled = {t: f for t, f in sorted(result.repeat_factor.items()) if f > 1}
    print(f"Oversampled types (copies): {oversampled}")
    print(f"Coverage {cov}")
    print("Split check: PASS (no shared patterns, no near-duplicate leaks)")
    print(f"Wrote {a.out_dir}, {a.manifest}, {a.data_card}")
```
- Print what was removed, which types were oversampled (`if f > 1` keeps only those with more than one copy), the coverage, the PASS message (reached only if the checks passed), and where files were written.

```python
    if not a.no_log:
        append_run_log(a.run_log, a.machine, stats, str(a.manifest.relative_to(REPO_ROOT)).replace("\\", "/"),
                       cfg, len(held["val"]), len(held["test"]))
        print(f"Appended an entry to {a.run_log.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```
- Same ending as Day 4: unless `--no-log`, append to `run_log.md`, return `0`; run `main()` only when the file is run directly and use its result as the exit code.

---

## 11. The whole run in order

1. Read options → absolute paths → build `SplitConfig`.
2. Check the two required inputs exist.
3. `load_questions` (validate) + `load_patterns` → cross-check IDs and types.
4. `choose_held_out_patterns` → `assignment`.
5. `build_splits` → near-duplicate guard → cap → oversample train → shuffle.
6. Write `train/val/test .txt` and `_labels.csv`; hash them.
7. Statistics, per-type table, coverage.
8. Write `split_manifest.json` and `docs/DATA_CARD.md`.
9. Leak check → console table → append to `run_log.md`.

## 12. How to read the results

- **`Split check: PASS`** — no pattern appears in two piles and no val/test sentence has the same wording as a train sentence.
- **Held-out patterns count** — about 7% of patterns each for val and test (fewer if few types are eligible; look for the WARNING).
- **Train `lines` vs `unique`** — `lines` is larger because of oversampling; `unique` is the real number of different questions.
- **Token bounds** — the real token count will be somewhere between the lower number and the UTF-8 byte count; Day 7 measures it.
- **Coverage below 100%** — normal: many glossary words are not used in questions.
- **Machine comparison** — compare `outputs_sha256` in the manifest between school and home. Same values = byte-identical outputs.

## 13. Small things worth knowing

- **Good improvement over Day 4:** `load_questions` skips bad rows with `continue`, so a row with the wrong number of columns is reported cleanly instead of crashing.
- **Leak check runs after writing.** If the check fails, the files are already on disk (the manifest and data card too). Treat them as invalid and fix the cause before using them.
- **`relative_to(REPO_ROOT)`** (same caveat as Day 4): if you point `--out-dir` or `--manifest` outside the repo, this call raises an error.
- **A label that is valid JSON but not an object** (for example `123` or a list) would make `label.get(...)` raise a Python `AttributeError` instead of the friendly message. The team's generator should never produce this.
- **Fixed rule text in the data card.** Numbers come from the run, but sentences like "at least 3 phrasings" and "at least 2 phrasings in train" are typed text that match today's rule in the code. If you change those rules in the code, update the card text too.
- **Rounding:** Python's `round` rounds exact halves to the nearest even number (e.g. `round(2.5)` is `2`), so the held-out counts can differ by one from what you expect by hand. It is still repeatable.

## 14. Quick summary in one paragraph

The script loads and validates the team's questions, holds out **whole sentence patterns** (one per chosen question type) for validation and test so the model is graded on wording it has never seen, removes near-copies and caps big held-out patterns, repeats small question types a few times in training only, writes plain question-only text files plus separate label files, and records everything (settings, fingerprints, statistics, coverage) in a manifest, a data card and the run log, with a final check that nothing leaked between the piles.
