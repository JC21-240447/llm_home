# day4_generate_corpus.py — Simple Line-by-Line Explanation (English)

## 0. The big picture (read this first)

Think of this script as a **quality-control manager in a factory**.

The factory machine already exists: it is the team's `make_questions.py`. That machine takes sentence patterns and word lists and produces thousands of Japanese questions. This script does **not** make questions itself. It **runs the machine safely, inspects the products, compares them with the team's master sample, and writes an inspection report**.

| Factory job | What this script does |
|---|---|
| Check the raw materials are in the warehouse | `check_inputs` — all input files and libraries exist |
| Move materials to a clean workbench (so the warehouse is never touched) | `prepare_work_dir` — copy inputs to a fresh timestamped folder |
| Start the machine with fixed settings | `run_generator` — same seed + same limit = same output every time |
| Inspect every product | `check_corpus` — no duplicates, no broken sentences, labels OK, 60 types |
| Compare with the master sample | `compare_with_team` / `compare_summary_files` |
| Write the inspection report | `write_report` + `append_run_log` |

Run it like this:

```
.venv\Scripts\python.exe scripts\day4_generate_corpus.py --machine home
```

**Why a work folder?** The team's generator writes its output files into the folder where it runs. If we ran it inside `data/source`, it would overwrite the team's own files. So we copy everything to a new folder first and run it there.

---

## 1. Words you need to know (mini dictionary)

| Word | Simple meaning |
|---|---|
| **Corpus** | A big collection of text. Here: thousands of generated questions. |
| **Pattern (文型)** | A sentence template with blanks, like "{社員}の{給与}を教えて". The generator fills the blanks with words from the glossary. |
| **Question type** | A group of patterns that ask the same kind of thing (this project expects 60 types). |
| **Seed** | A starting number for the random generator. Same seed = same "random" choices every time. This makes results **repeatable**. |
| **Per-pattern limit** | Max questions kept per pattern (2000). Stops one pattern with millions of combinations from dominating the corpus (called *balancing*). |
| **SHA-256 hash** | A fingerprint of a file: a long string computed from its bytes. Change one byte → totally different fingerprint. Used to prove which exact input files were used. |
| **Byte-identical** | Two files match in every single byte. The strictest "same". |
| **CSV** | A plain-text table: rows on lines, columns separated by commas. |
| **BOM / `utf-8-sig`** | A tiny invisible marker some programs (Excel) put at the start of a UTF-8 file. `utf-8-sig` reads the file and removes that marker. |
| **ZIP** | A compressed archive holding one or more files. |
| **`dataclass`** | A short way to make a class that just holds data. |
| **Exit code** | The number a program returns to the system: 0 = OK, non-zero = problem. |
| **`SystemExit`** | A Python exception that stops the program on purpose (with an optional message). |
| **cwd** | "Current working directory": the folder a program thinks it is running in. |

---

## 2. The docstring (top of the file)

```python
r"""Day 4 - generate the question corpus ...
...
"""
```
- Same idea as Day 1: a raw (`r`) triple-quoted documentation string, English part + Japanese part, never executed.
- It lists the 6 steps of the script (check, copy, run, inspect, compare, report), the usage command, and a note: on the school PC use `--machine school`, and make sure `pandas` and `openpyxl` are installed there too.
- "CPU only" — unlike Day 1, nothing here uses the GPU.

---

## 3. Imports

```python
from __future__ import annotations
```
- Same as Day 1 (modern type hints work everywhere).

```python
import argparse
import csv
import hashlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
```

`argparse`, `json`, `os`, `shutil`, `subprocess`, `sys`, `time`, `datetime`, `Path` — **same as Day 1** (command-line options, JSON, file copying, running other programs, stopwatch, dates, paths). New ones:

| Line | What it gives you |
|---|---|
| `import csv` | Read CSV files properly (handles commas and quotes inside text). |
| `import hashlib` | Computes fingerprints (SHA-256). |
| `import importlib.util` | Lets us ask "is this library installed?" without importing it. |
| `import io` | Lets us treat a string in memory like a file (`StringIO`). |
| `import zipfile` | Opens `.zip` archives. |
| `from collections import Counter, defaultdict` | `Counter` counts things. `defaultdict` is a dictionary that creates a missing entry automatically. |
| `from dataclasses import dataclass` | The shortcut for data-holding classes. |

---

## 4. Constants

```python
SCRIPT_VERSION: str = "day4-v1"
```
- A version label written into the report, so you know which version of the script made it.

```python
REPO_ROOT: Path = Path(__file__).resolve().parent.parent
```
- Same as Day 1: the repo's root folder, found from this script's location.

```python
GENERATOR: str = "make_questions.py"
REQUIRED_INPUTS: tuple[str, ...] = (GENERATOR, "用語集.csv", "同義語辞書.csv", "DB.xlsx")
```
- `GENERATOR` — the file name of the team's generator.
- `REQUIRED_INPUTS` — a **tuple** (a list that cannot be changed) of the four files that must exist: the generator, the glossary (用語集), the synonym dictionary (同義語辞書), and the sample table `DB.xlsx`. `tuple[str, ...]` means "any number of strings".

```python
SUMMARY_FILES: tuple[str, ...] = ("patterns.csv", "question_types.csv", "columns.csv")
```
- Three summary files the generator creates. We later compare them with the team's copies.

```python
QUESTION_HEADER: list[str] = ["質問", "パターンID", "質問タイプID", "意味"]
```
- The exact first row (column names) that `questions.csv` must have: question, pattern ID, question type ID, meaning (label).

```python
BROKEN_MARKS: tuple[str, ...] = ("{", "}", "のの", "社員の社員", "平均平均")
```
- Signs of a **broken sentence**:
  - `{` or `}` — a blank in the pattern was never filled.
  - `のの`, `社員の社員`, `平均平均` — words glued together wrongly (a doubled "の", etc.).

---

## 5. The data class

```python
@dataclass
class GeneratedCorpus:
    """The generator's questions.csv, parsed."""

    raw_bytes: bytes
    rows: list[list[str]]  # 質問, パターンID, 質問タイプID, 意味
```
- `@dataclass` is a **decorator**: it automatically writes the boring parts of a class (the `__init__` that stores the values).
- So `GeneratedCorpus(raw, rows)` makes a box with two things:
  - `raw_bytes` — the original file bytes (kept so we can compare byte-by-byte and fingerprint later).
  - `rows` — the table as a list of rows; each row is a list of 4 strings.

---

## 6. Helper functions

### 6.1 Fingerprints

```python
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())
```
- `sha256_bytes` — computes the SHA-256 of some bytes and returns it as a hex string (letters and digits).
- `sha256_file` — reads a whole file as bytes (`read_bytes`) and calls `sha256_bytes`. (Same job, but for a file.)

### 6.2 `check_inputs`

```python
def check_inputs(source_dir: Path) -> list[Path]:
    """Stop with a clear message if an input file or library is missing. Returns the src_*.txt files."""
    missing = [n for n in REQUIRED_INPUTS if not (source_dir / n).exists()]
    src_files = sorted(source_dir.glob("src_*.txt"))
    if not src_files:
        missing.append("src_*.txt")
```
- `missing = [...]` — a **list comprehension**: go through the 4 required names and keep those that do **not** exist in `source_dir`.
- `source_dir.glob("src_*.txt")` — find all files whose name starts with `src_` and ends with `.txt` (the sentence-pattern files). `sorted(...)` puts them in a fixed order, so results are repeatable.
- If there are none, add `"src_*.txt"` to the missing list.

```python
    if missing:
        raise SystemExit(
            f"Missing in {source_dir}: {', '.join(missing)}\n"
            "Put the team's files there first (see Step 1 of the Day 4 walkthrough)."
        )
```
- If anything is missing, stop the whole program with a clear message listing what is missing. `', '.join(missing)` turns the list into `a, b, c`.

```python
    libs = [m for m in ("pandas", "openpyxl") if importlib.util.find_spec(m) is None]
    if libs:
        raise SystemExit(
            f"Missing Python libraries: {', '.join(libs)}\n"
            r"Install them: .venv\Scripts\python.exe -m pip install -r requirements.txt"
        )
    return src_files
```
- `find_spec(m)` asks Python "can you find library `m`?" It returns `None` if not. So `libs` = the libraries that are missing.
- `pandas` and `openpyxl` are needed by the **generator** (to read `DB.xlsx`), not by this script itself. We check now so the failure is early and clear.
- The `r"..."` raw string keeps the backslashes in the Windows path as-is.
- At the end, return the list of pattern files.

### 6.3 `prepare_work_dir`

```python
def prepare_work_dir(source_dir: Path, work_root: Path, src_files: list[Path]) -> Path:
    """Copy the generator and its inputs into a fresh folder, so data/source is never written to."""
    work = work_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    work.mkdir(parents=True, exist_ok=False)
    for name in REQUIRED_INPUTS:
        shutil.copy2(source_dir / name, work / name)
    for f in src_files:
        shutil.copy2(f, work / f.name)
    return work
```
- `work = work_root / <timestamp>` — a new folder named like `20261106_143012` (date + time to the second), so each run gets its own folder.
- `work.mkdir(parents=True, exist_ok=False)` — create it. `exist_ok=False` is deliberate: if the folder somehow already exists, **raise an error** instead of reusing it. "Fresh" means truly fresh.
- First loop — copy the 4 required files into the work folder. `shutil.copy2` copies the file **and** its metadata (like modified time).
- Second loop — copy every `src_*.txt` pattern file. `f.name` is just the file name without the folder.
- Return the work folder path.

### 6.4 `run_generator`

```python
def run_generator(work: Path, per_pattern: int, seed: int) -> tuple[str, float]:
    """Run the team's generator in the work folder with this venv's Python. Returns (console output, seconds)."""
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
```
- `os.environ` is the set of environment variables. `dict(os.environ, A=..., B=...)` makes a **copy** and adds/overrides two settings. They force Python to use UTF-8 everywhere in the child program, so Japanese does not break on Windows.

```python
    cmd = [sys.executable, GENERATOR, "--corpus", str(per_pattern), "--seed", str(seed)]
```
- Builds the command as a list: `python make_questions.py --corpus 2000 --seed 42`.
- `sys.executable` = the exact Python running this script (the venv's Python), so the generator uses the same libraries.
- `str(...)` turns numbers into text, because command-line arguments must be strings.

```python
    start = time.perf_counter()
    proc = subprocess.run(cmd, cwd=work, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    seconds = time.perf_counter() - start
```
- `perf_counter()` — a high-precision stopwatch.
- `subprocess.run(...)` — runs the command:
  - `cwd=work` — run **inside** the work folder (this is why the outputs land there).
  - `env=env` — use the UTF-8 settings above.
  - `capture_output=True`, `text=True` — same as Day 1: catch the output as text.
  - `encoding="utf-8", errors="replace"` — decode the output as UTF-8; unreadable bytes become `?` instead of crashing.
- `seconds` = end time minus start time.

```python
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr)
        raise SystemExit(f"The generator stopped with exit code {proc.returncode}. See the error above.")
    return proc.stdout, seconds
```
- If the generator failed (non-zero exit code), print everything it said (normal output and error output) so you can see why, then stop.
- Otherwise return its console output and the time taken.

### 6.5 `read_corpus_bytes`

```python
def read_corpus_bytes(path: Path) -> bytes:
    """questions.csv bytes, from a .csv or from the questions.csv inside a .zip."""
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            members = [n for n in zf.namelist() if n.endswith("questions.csv")]
            if len(members) != 1:
                raise SystemExit(f"{path}: expected one questions.csv inside, found {members}")
            return zf.read(members[0])
    return path.read_bytes()
```
- `path.suffix.lower() == ".zip"` — is the file a ZIP (ignoring upper/lower case)?
- `zipfile.ZipFile(path)` opens it (`with` closes it afterward).
- `zf.namelist()` lists the files inside. We keep those ending in `questions.csv`.
- It must be **exactly one**. Zero or many → stop with a message showing what was found.
- `zf.read(members[0])` reads that file's bytes **without** unpacking to disk.
- If it is not a ZIP, just read the file's bytes directly.

### 6.6 `parse_corpus`

```python
def parse_corpus(raw: bytes) -> GeneratedCorpus:
    reader = csv.reader(io.StringIO(raw.decode("utf-8-sig")))
    header = next(reader)
    if header != QUESTION_HEADER:
        raise SystemExit(f"Unexpected header {header}")
    return GeneratedCorpus(raw, [row for row in reader])
```
- `raw.decode("utf-8-sig")` — turn bytes into text and drop the BOM if present.
- `io.StringIO(...)` — wrap the text so it behaves like an open file.
- `csv.reader(...)` — reads it row by row, as lists of strings.
- `header = next(reader)` — take the **first** row (the column names).
- If the header is not exactly `QUESTION_HEADER`, stop: the file is not in the expected format.
- `[row for row in reader]` — collect the remaining rows into a list.
- Return the dataclass with both the raw bytes and the parsed rows.

---

## 7. The quality checks: `check_corpus`

```python
def check_corpus(corpus: GeneratedCorpus, expected_types: int) -> dict[str, object]:
    """Quality checks on the generated questions. Raises SystemExit if any check fails."""
    texts = [r[0] for r in corpus.rows]
    problems: list[str] = []
```
- `texts` — just the question sentences (column 0 of every row).
- `problems` — an empty list where we collect descriptions of every problem found. (Collect all first, report all together.)

```python
    if any(len(r) != 4 for r in corpus.rows):
        problems.append("a row does not have 4 columns")
```
- **Check 1: shape.** `any(...)` = "is at least one row not exactly 4 columns?"

```python
    if len(set(texts)) != len(texts):
        problems.append(f"{len(texts) - len(set(texts))} duplicate questions")
```
- **Check 2: duplicates.** A `set` removes duplicates. If the set is shorter than the list, some questions repeated. The difference = how many.

```python
    broken = [t for t in texts if any(m in t for m in BROKEN_MARKS)]
    if broken:
        problems.append(f"{len(broken)} broken sentences, e.g. {broken[0]}")
```
- **Check 3: broken sentences.** Keep every question that contains **any** of the bad marks. If there are some, report how many plus one example (`broken[0]`).

```python
    bad_labels = 0
    for text, pid, tid, meaning in corpus.rows:
        try:
            label = json.loads(meaning)
            if label.get("パターン") != pid or label.get("質問タイプ") != tid:
                bad_labels += 1
        except json.JSONDecodeError:
            bad_labels += 1
    if bad_labels:
        problems.append(f"{bad_labels} labels unreadable or not matching their row")
```
- **Check 4: labels.** Each row's 4th column (`meaning`) is a small JSON text. For each row:
  - `for text, pid, tid, meaning in ...` — unpack the 4 columns into 4 names.
  - `json.loads(meaning)` — turn the JSON text into a dictionary.
  - Compare the pattern and type stored **inside** the label with the pattern ID and type ID stored in the row's own columns. Different → count as bad.
  - If the JSON itself is broken (`JSONDecodeError`) → also bad.
- Report the count if above zero.

```python
    types = {r[2] for r in corpus.rows}
    if len(types) != expected_types:
        problems.append(f"{len(types)} question types, expected {expected_types}")
```
- **Check 5: question types.** `{ ... }` with a loop is a **set comprehension**: the distinct type IDs. There must be exactly `expected_types` (60).

```python
    if problems:
        raise SystemExit("Corpus check FAILED: " + "; ".join(problems))
```
- If any check found anything, stop the program and show all problems joined with `; `.

```python
    lengths = [len(t) for t in texts]
    return {
        "questions": len(texts),
        "patterns_with_questions": len({r[1] for r in corpus.rows}),
        "types": len(types),
        "characters": sum(lengths),
        "avg_chars": round(sum(lengths) / len(lengths), 2),
        "min_chars": min(lengths),
        "max_chars": max(lengths),
    }
```
- If everything passed, compute statistics: `lengths` = number of characters of each question.
- Return a dictionary: total questions; how many different patterns appear; number of types; total characters; average, shortest and longest question length.

---

## 8. Comparing with the team

### 8.1 `compare_with_team`

```python
def compare_with_team(new: GeneratedCorpus, team_raw: bytes | None) -> dict[str, object]:
    """Byte comparison plus a per-pattern comparison of the question sets."""
    if team_raw is None:
        return {"available": False}
    team = parse_corpus(team_raw)
```
- `team_raw` may be `None` if the team's corpus file was not found. Then we just report "not available".
- Otherwise parse the team's corpus the same way as ours.

```python
    def by_pattern(c: GeneratedCorpus) -> dict[str, set[str]]:
        d: dict[str, set[str]] = defaultdict(set)
        for text, pid, _, _ in c.rows:
            d[pid].add(text)
        return d
```
- A **function inside a function** (used only here).
- `defaultdict(set)` — a dictionary where a missing key automatically starts as an empty set.
- For each row, add the question text to the set belonging to its pattern ID. Result: `{pattern_id: {all questions of that pattern}}`.
- `_` means "I don't need this column".

```python
    a, b = by_pattern(new), by_pattern(team)
    all_pids = sorted(set(a) | set(b))
    differing = [p for p in all_pids if a.get(p) != b.get(p)]
```
- `a` = ours, `b` = the team's.
- `set(a) | set(b)` — the **union** of pattern IDs from both (so a pattern that exists on only one side is still checked). `sorted` for stable order.
- `differing` — patterns where our set of questions is **not equal** to the team's set. Order does not matter for sets; only the content.

```python
    return {
        "available": True,
        "byte_identical": new.raw_bytes == team_raw,
        "team_questions": len(team.rows),
        "patterns_identical": len(all_pids) - len(differing),
        "patterns_differing": differing,
    }
```
- `byte_identical` — are the two files the same in every byte? (the strictest test)
- `team_questions` — how many questions the team's corpus has.
- `patterns_identical` — total patterns minus differing ones.
- `patterns_differing` — the list of the differing pattern IDs.

### 8.2 `compare_summary_files`

```python
def compare_summary_files(work: Path, source_dir: Path) -> dict[str, str]:
    """Is each regenerated summary file identical to the team's copy in data/source?"""
    out: dict[str, str] = {}
    for name in SUMMARY_FILES:
        team = source_dir / name
        if not team.exists():
            out[name] = "no team copy"
        else:
            out[name] = "identical" if (work / name).read_bytes() == team.read_bytes() else "DIFFERENT"
    return out
```
- For each of `patterns.csv`, `question_types.csv`, `columns.csv`:
  - If the team has no copy → `"no team copy"`.
  - Otherwise compare the freshly generated file (in the work folder) with the team's file, byte by byte → `"identical"` or `"DIFFERENT"`.

---

## 9. The report

### 9.1 `load_pattern_table`

```python
def load_pattern_table(work: Path) -> list[dict[str, str]]:
    with (work / "patterns.csv").open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))
```
- Opens the generated `patterns.csv` (removing a BOM if there is one).
- `newline=""` is what the `csv` module wants so line breaks inside quoted cells are handled properly.
- `csv.DictReader` turns each row into a dictionary keyed by the column names (e.g. `p["パターンID"]`). `list(...)` collects them all.

### 9.2 `write_report` — the data preparation part

```python
def write_report(path, stats, cmp_team, cmp_files, patterns, corpus, inputs, per_pattern, seed, seconds, generator_out) -> None:
```
- Takes everything we learned so far and writes one Markdown file. (The parameters are written one per line in the real code for readability.)

```python
    gen_count = Counter(r[1] for r in corpus.rows)
```
- `Counter` counts how many generated questions each pattern ID has.

```python
    type_rows: dict[str, list] = {}
    for p in patterns:
        t = type_rows.setdefault(p["質問タイプID"], [p["質問タイプ"], 0, 0, 0])
        t[1] += 1
        t[2] += int(p["作れる質問の数"])
        t[3] += gen_count.get(p["パターンID"], 0)
```
- Builds a summary **per question type**. Each entry is a list: `[type name, number of patterns, possible questions, written questions]`.
- `setdefault(key, default)` — "give me the entry for this key; if there is none yet, create it with this default." The default has the type name and three zeros.
- `t[1] += 1` — one more pattern belongs to this type.
- `t[2] += int(...)` — add how many questions this pattern **could** make (the CSV stores numbers as text, so `int()`).
- `t[3] += gen_count.get(..., 0)` — add how many questions were **actually written** for this pattern (0 if none).

```python
    first_question: dict[str, str] = {}
    for text, _, tid, _ in corpus.rows:
        first_question.setdefault(tid, text)
```
- For each type, remember the **first** question we meet (`setdefault` only stores if nothing is there yet). Used as the "example" column.

```python
    styles = Counter(p["文体"] for p in patterns)
    empty = [p["パターンID"] for p in patterns if gen_count.get(p["パターンID"], 0) == 0]
    possible = sum(int(p["作れる質問の数"]) for p in patterns)
```
- `styles` — how many patterns per writing style (文体), e.g. polite vs casual.
- `empty` — patterns that ended with **zero** questions (all removed as duplicates, for example).
- `possible` — the total number of distinct questions all patterns could make together.

### 9.3 `write_report` — building the Markdown

```python
    lines = [
        "# Day 4 report: question generation",
        "",
        f"Generated by `scripts/day4_generate_corpus.py` ({SCRIPT_VERSION}) on {datetime.now():%Y-%m-%d %H:%M}.",
        ...
```
- `lines` is a list of text lines that will become the report. `""` = blank line.
- `{datetime.now():%Y-%m-%d %H:%M}` — a date formatted right inside the f-string (the part after `:` is the format).
- `` `...` `` backticks are Markdown for code style.

```python
        "| File | SHA-256 |",
        "| --- | --- |",
        *(f"| `{n}` | `{h}` |" for n, h in inputs.items()),
```
- The first two lines are the header of a Markdown table.
- `*( ... )` **unpacks** a generator into the list: one table row per input file with its fingerprint. This is the "which exact input files produced this corpus" proof.

```python
        f"- At most {per_pattern:,} questions per pattern (balancing), seed {seed}.",
        f"- Generation time: {seconds:.1f} seconds.",
```
- `{per_pattern:,}` — a number with thousands separators (`2,000`). `{seconds:.1f}` — one decimal place.

```python
        f"| Patterns | {len(patterns)} ({', '.join(f'{k} {v}' for k, v in styles.items())}) |",
        f"| Question types | {stats['types']} |",
        f"| All distinct questions the patterns can make | {possible:,} |",
        f"| Questions written (after the per-pattern limit and duplicate removal) | {stats['questions']:,} |",
        f"| Patterns with at least one question | {stats['patterns_with_questions']} |",
        f"| Patterns with no question left after duplicate removal | {', '.join(empty) or 'none'} |",
        f"| Characters | {stats['characters']:,} |",
        f"| Length per question (min / average / max) | {stats['min_chars']} / {stats['avg_chars']} / {stats['max_chars']} |",
```
- Rows of the **Result** table. Each line is "a label | a value".
- First row: number of patterns, then a list like `polite 40, casual 20` made with a nested f-string (note the different quote marks inside).
- Other rows pull numbers from `stats` and format them with `:,`.
- `', '.join(empty) or 'none'` — join the empty-pattern IDs; if the result is an empty string, show `none` (`or` picks the second value when the first is empty).

```python
        "Checks passed: no duplicates, no unfilled slots or broken joins, every row has a readable label",
        "that matches its pattern and type.",
```
- A fixed sentence. It is safe to always print "passed" because failures stop the program earlier.

```python
    if cmp_team.get("available"):
        lines += [
            f"- Compared with the team's corpus (`questions_corpus.zip`, {cmp_team['team_questions']:,} questions):",
            f"  byte-identical: **{'yes' if cmp_team['byte_identical'] else 'no'}**;"
            f" identical patterns: {cmp_team['patterns_identical']};"
            f" differing patterns: {', '.join(cmp_team['patterns_differing']) or 'none'}.",
        ]
    else:
        lines.append("- The team's corpus was not found, so no comparison was made.")
```
- If the team's corpus was available, add two lines about the comparison (the `**...**` is Markdown bold; the split strings are auto-joined into one line). If not, add a one-line note.

```python
    lines += [f"- `{n}` compared with the team's copy: {v}." for n, v in cmp_files.items()]
```
- One line per summary file: `identical`, `DIFFERENT`, or `no team copy`.

```python
        *(
            f"| {tid} | {v[0]} | {v[1]} | {v[2]:,} | {v[3]:,} | {first_question.get(tid, '')} |"
            for tid, v in sorted(type_rows.items())
        ),
```
- The **per-type table** rows (after its header lines): type ID, name, number of patterns, possible, written, and the example. `sorted(...)` orders by type ID.

```python
        "```text",
        generator_out.strip(),
        "```",
```
- Adds the generator's own console output inside a Markdown code block.

```python
    path.write_text("\n".join(lines), encoding="utf-8")
```
- Join all lines with line breaks and save as UTF-8.

### 9.4 `append_run_log`

```python
def append_run_log(log: Path, machine: str, stats, cmp_team, per_pattern: int, seed: int, seconds: float, work_rel: str) -> None:
    same = "n/a" if not cmp_team.get("available") else ("yes" if cmp_team["byte_identical"] else
                                                        f"no ({len(cmp_team['patterns_differing'])} patterns differ)")
```
- Builds one short text `same`:
  - team corpus not available → `n/a`
  - byte-identical → `yes`
  - otherwise → `no (N patterns differ)`.
- This is a **nested conditional expression** (an if/else inside an if/else).

```python
    entry = [
        "",
        f"## {datetime.now():%Y-%m-%d %H:%M} | {machine} | Day 4 question generation",
        "",
        f"- Machine label: {machine} (CPU only; no GPU used)",
        f"- Generator: data/source/make_questions.py (team), run by scripts/day4_generate_corpus.py ({SCRIPT_VERSION})",
        f"- Settings: max {per_pattern} per pattern, seed {seed}; time {seconds:.1f} s",
        f"- Result: {stats['questions']:,} questions, {stats['types']} types, "
        f"{stats['patterns_with_questions']} patterns with questions, avg {stats['avg_chars']} chars",
        f"- Identical to the team's corpus: {same}",
        f"- Output folder: {work_rel}",
        "- Checks: PASS",
        "",
    ]
    with log.open("a", encoding="utf-8") as f:
        f.write("\n".join(entry))
```
- Builds a short Markdown log entry (same style as Day 1's `append_run_log`).
- `"- Checks: PASS"` is always written, because any failed check would already have stopped the program.
- Opens `run_log.md` in append mode `"a"` and adds the entry at the end.

---

## 10. Main part

### 10.1 `parse_args`

```python
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Day 4: run the team's question generator, check and report.")
    ap.add_argument("--machine", choices=["school", "home"], required=True)
    ap.add_argument("--source-dir", type=Path, default=REPO_ROOT / "data/source")
    ap.add_argument("--work-root", type=Path, default=REPO_ROOT / "runs/day4_generation")
    ap.add_argument("--team-corpus", type=Path, default=REPO_ROOT / "data/source/questions_corpus.zip")
    ap.add_argument("--report", type=Path, default=REPO_ROOT / "docs/DAY4_GENERATION_REPORT.md")
    ap.add_argument("--run-log", type=Path, default=REPO_ROOT / "run_log.md")
    ap.add_argument("--per-pattern", type=int, default=2000, help="max questions per pattern (team setting: 2000)")
    ap.add_argument("--seed", type=int, default=42, help="random seed (team setting: 42)")
    ap.add_argument("--expected-types", type=int, default=60)
    ap.add_argument("--no-log", action="store_true")
    return ap.parse_args()
```
- Same `argparse` idea as Day 1. `ap` is the short name for the parser.
- `--machine` — required, `school` or `home` (same as Day 1).
- `--source-dir`, `--work-root`, `--team-corpus`, `--report`, `--run-log` — five **path** options. `type=Path` converts what you type into a `Path`. All have defaults, so normally you type none of them. (These five lines all follow the same pattern; only the name and the default location differ.)
- `--per-pattern` — a whole number (`type=int`), default 2000 (the team's setting). `help=` is the text shown with `--help`.
- `--seed` — default 42 (the team's setting).
- `--expected-types` — how many question types the check expects (60).
- `--no-log` — a flag (`store_true`, as in Day 1): if written, do **not** add an entry to `run_log.md`.
- `ap.parse_args()` reads what you typed and returns it.

### 10.2 `main` — start

```python
def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
```
- If the output stream supports `reconfigure`, tell it to replace unprintable characters instead of crashing. (A lighter version of Day 1's `setup_utf8_console`; here only `errors` is set.)

```python
    a = parse_args()
    for name in ("source_dir", "work_root", "team_corpus", "report", "run_log"):
        setattr(a, name, getattr(a, name).resolve())
```
- `a` holds all the options (`a.machine`, `a.seed`, ...).
- The loop turns each of the 5 path options into a **full absolute path** with `.resolve()`. `getattr(a, name)` = read the attribute whose name is in the text; `setattr(a, name, value)` = write it back. This avoids repeating the same line five times.

```python
    print(f"=== Day 4 question generation | machine={a.machine} | max {a.per_pattern}/pattern | seed {a.seed} ===")
    src_files = check_inputs(a.source_dir)
    work = prepare_work_dir(a.source_dir, a.work_root, src_files)
    print(f"Work folder: {work}")
```
- Header → step 1 (check inputs) → step 2 (copy into a fresh work folder) → show where the work folder is.

```python
    generator_out, seconds = run_generator(work, a.per_pattern, a.seed)
    print(generator_out.strip())
    print(f"Generation took {seconds:.1f} s")
```
- Step 3: run the team's generator. Get back its output and the time. Print both.

```python
    corpus = parse_corpus((work / "questions.csv").read_bytes())
    stats = check_corpus(corpus, a.expected_types)
    team_raw = read_corpus_bytes(a.team_corpus) if a.team_corpus.exists() else None
    cmp_team = compare_with_team(corpus, team_raw)
    cmp_files = compare_summary_files(work, a.source_dir)
```
- Step 4: read the generated `questions.csv` and parse it → run the 5 quality checks.
- Step 5: read the team's corpus **only if it exists** (else `None`) → compare questions → compare summary files.

```python
    inputs = {f.name: sha256_file(f) for f in [*(a.source_dir / n for n in REQUIRED_INPUTS), *src_files]}
```
- A **dictionary comprehension**: file name → fingerprint.
- The list being looped over is built by joining two things with `*`: the 4 required files (full paths made with `a.source_dir / n`) and the `src_*.txt` files.

```python
    a.report.parent.mkdir(parents=True, exist_ok=True)
    write_report(a.report, stats, cmp_team, cmp_files, load_pattern_table(work), corpus, inputs,
                 a.per_pattern, a.seed, seconds, generator_out)
```
- Make sure the report's folder (`docs/`) exists, then write the report. `load_pattern_table(work)` supplies the pattern table.

```python
    (work / "check_result.json").write_text(
        json.dumps({"stats": stats, "compare_team": cmp_team, "compare_files": cmp_files,
                    "questions_csv_sha256": sha256_bytes(corpus.raw_bytes)}, ensure_ascii=False, indent=2),
        encoding="utf-8")
```
- Also save a machine-readable `check_result.json` inside the work folder with the stats, the comparisons, and the fingerprint of the generated `questions.csv`. `ensure_ascii=False` keeps Japanese readable (same as Day 1), `indent=2` makes it pretty.

### 10.3 `main` — printing and finishing

```python
    print()
    print(f"Questions: {stats['questions']:,} | types: {stats['types']} | "
          f"patterns with questions: {stats['patterns_with_questions']} | avg chars: {stats['avg_chars']}")
    print("Corpus check: PASS (no duplicates, no broken sentences, labels readable)")
```
- A blank line, then a one-line summary of the numbers and a PASS message (we only get here if all checks passed).

```python
    if cmp_team.get("available"):
        print(f"Same as the team's corpus: byte-identical={cmp_team['byte_identical']}, "
              f"identical patterns={cmp_team['patterns_identical']}, "
              f"differing patterns={cmp_team['patterns_differing'] or 'none'}")
    for n, v in cmp_files.items():
        print(f"{n}: {v}")
    print(f"Report: {a.report}")
```
- Print the comparison result (if the team's corpus exists), one line per summary file, and where the report was saved.

```python
    if not a.no_log:
        work_rel = str(work.relative_to(REPO_ROOT)).replace("\\", "/")
        append_run_log(a.run_log, a.machine, stats, cmp_team, a.per_pattern, a.seed, seconds, work_rel)
        print(f"Appended an entry to {a.run_log.name}")
    return 0
```
- Unless `--no-log` was given: make the work folder path **relative to the repo** (`relative_to`) and switch Windows `\` to `/` so the log looks the same on every machine; append the log entry; print a confirmation.
- `return 0` — success.

```python
if __name__ == "__main__":
    sys.exit(main())
```
- Same idea as Day 1 (`SystemExit(main())`): run `main()` only when the file is run directly, and use its return value as the exit code. `sys.exit(...)` and `raise SystemExit(...)` do the same thing.
- Remember: all the `raise SystemExit("message")` calls earlier also stop the program, print the message, and give exit code 1.

---

## 11. The whole run in order

1. Read options → turn paths into absolute paths.
2. `check_inputs` — files and libraries exist?
3. `prepare_work_dir` — copy into `runs/day4_generation/<time>/`.
4. `run_generator` — run `make_questions.py` there with seed 42, limit 2000.
5. `parse_corpus` → `check_corpus` — 5 quality checks.
6. Compare with the team's corpus and summary files.
7. Hash the inputs → write `docs/DAY4_GENERATION_REPORT.md` + `check_result.json`.
8. Print the summary → append to `run_log.md` (unless `--no-log`).

## 12. How to read the results

- **`byte-identical: yes`** — your run reproduced the team's corpus exactly. Best possible result.
- **`byte-identical: no` + a list of differing patterns** — something differs. Look at those pattern IDs first (different input files, different library versions, or a different seed or limit).
- **Summary files `identical`** — the generator also reproduced the team's `patterns.csv`, `question_types.csv`, `columns.csv`.
- **"Patterns with no question left after duplicate removal"** — patterns whose questions were all duplicates of other patterns' questions.
- **Corpus check FAILED** — read the message: it lists every problem found (duplicates, broken sentences, bad labels, wrong type count).

## 13. Small things worth knowing (not bugs in normal use)

- **Only checks that fail loudly.** All failures use `SystemExit`, so a failed run never writes "PASS" in the log.
- **4-column check order.** If a row has fewer or more than 4 columns, the later label loop (which unpacks 4 values) would crash with a Python `ValueError` before the friendly message is shown. With the team's generator this should not happen; it only matters if the CSV is damaged.
- **`--work-root` outside the repo.** The final `relative_to(REPO_ROOT)` would raise an error if you pointed `--work-root` to a folder outside the repo. With the default it is fine.
- **Same seed ≠ same output on different libraries.** Reproducibility depends on the same inputs, same seed, same limit — and also the same generator code and library behaviour. That is why the report stores the input fingerprints.

## 14. Quick summary in one paragraph

The script runs the team's question generator in a clean copy of the inputs with a fixed seed, checks the result for duplicates, broken sentences, unreadable labels and the right number of question types, compares it byte-by-byte and pattern-by-pattern with the team's corpus, and then writes a Markdown report (with fingerprints of every input), a JSON result, and a note in `run_log.md`, so the generation is safe, repeatable and provable.
