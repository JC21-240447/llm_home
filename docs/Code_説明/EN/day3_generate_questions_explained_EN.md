# Day 3 — day3_generate_questions.py — Simple Line-by-Line Explanation (English)

> **How the two Day 3 files fit together**
> - **Day 3.1 `make_questions.py`** — the team's question generator (the machine).
> - **Day 3 (this file) `day3_generate_questions.py`** — the *runner*: runs the generator safely with the team's settings, checks and saves the result (the run shown on video).
>
> Read **Day 3.1 first** if you want to know *what* is being run. This file explains *how it is run*.
> Day 4 (already explained) reuses several functions from this script — those parts are marked "same as Day 4".

---

## 0. The big picture (read this first)

Think of a **factory supervisor** (this script) and a **machine** (`make_questions.py`).

The supervisor:
1. checks that all the raw materials and tools are there;
2. moves the materials to a **fresh workbench** (so the original files are never touched);
3. starts the machine with **fixed settings** (at most 2,000 questions per pattern, seed 42) and times it;
4. **inspects** the products (no duplicates, no unfilled slots, 60 question types);
5. saves the products, a **zip to carry to the other PC**, and a **receipt** (`generation_info.json`) with fingerprints;
6. prints a summary and **10 example questions**, and writes a note in `run_log.md`.

Day 4 later regenerates the corpus (on another machine) and compares it with this result.

Run it:

```
.venv\Scripts\python.exe scripts\day3_generate_questions.py --machine home
```

---

## 1. Words you need to know

All the words are the same as Days 4–5: **seed**, **SHA-256**, **work folder**, **round trip**, **zip**. New ones:

| Word | Simple meaning |
|---|---|
| **`generation_info.json`** | The "receipt" of the run: settings, input/output fingerprints, counts, time. |
| **Zip for the other PC** | `questions.csv` compressed into a zip that you can carry to another machine to compare. |
| **`random.sample`** | Pick several different items at random from a list (no repeats). |

---

## 2. The docstring

- English + Japanese, 6 steps, usage command, and a **school-PC note** with two important rules:
  - install `pandas` and `openpyxl` **into that PC's own `.venv` only** (never into the school's Python);
  - the school repository is **public**: **do not commit** `data/source/` or the generated files (they contain the team's table and internal labels).
- It says Day 4 will regenerate on another machine and compare with the zip.

---

## 3. Imports and constants

```python
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path
```
- All the same as Day 4 (see Day 4 for each). Two differences: `platform` (Python version, as in Days 6–7) and `random` (for the 10 examples). `Counter` and `dataclass` are not needed here.

```python
SCRIPT_VERSION: str = "day3-v1"
REPO_ROOT: Path = Path(__file__).resolve().parent.parent
GENERATOR: str = "make_questions.py"
REQUIRED_INPUTS: tuple[str, ...] = (GENERATOR, "用語集.csv", "同義語辞書.csv", "DB.xlsx")
QUESTION_HEADER: list[str] = ["質問", "パターンID", "質問タイプID", "意味"]
BROKEN_MARKS: tuple[str, ...] = ("{", "}", "のの", "社員の社員", "平均平均")
```
- Same as Day 4 (version name, repo root, generator file name, the four required files, the exact header, the marks of a broken sentence). There is no `SUMMARY_FILES` here.

---

## 4. Helper functions

### 4.1 `sha256_bytes`

```python
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
```
- Same as Day 4: fingerprint of some bytes.

### 4.2 `check_inputs` (same as Day 4)

```python
def check_inputs(source_dir: Path) -> list[Path]:
    missing = [n for n in REQUIRED_INPUTS if not (source_dir / n).exists()]
    src_files = sorted(source_dir.glob("src_*.txt"))
    if not src_files:
        missing.append("src_*.txt")
    if missing:
        raise SystemExit(f"Missing in {source_dir}: {', '.join(missing)}")
    libs = [m for m in ("pandas", "openpyxl") if importlib.util.find_spec(m) is None]
    if libs:
        raise SystemExit(...)
    return src_files
```
- Same logic as Day 4's `check_inputs`: collect missing files (the four required + at least one `src_*.txt`), stop if any; check that `pandas` and `openpyxl` can be found, stop if not; return the pattern files.
- The library message here tells you to install **into this `.venv` only**: `.venv\Scripts\python.exe -m pip install pandas openpyxl`.

### 4.3 `copy_inputs` (same idea as Day 4's `prepare_work_dir`)

```python
def copy_inputs(source_dir: Path, out_dir: Path, src_files: list[Path]) -> None:
    out_dir.mkdir(parents=True, exist_ok=False)
    for name in REQUIRED_INPUTS:
        shutil.copy2(source_dir / name, out_dir / name)
    for f in src_files:
        shutil.copy2(f, out_dir / f.name)
```
- Creates the **new** output folder (`exist_ok=False` → error if it already exists, so it is really fresh) and copies the generator, the inputs and the pattern files into it. The difference from Day 4: this function **receives** the folder path (`main` decides the time-stamped name).

### 4.4 `run_generator` (same as Day 4)

```python
def run_generator(out_dir: Path, per_pattern: int, seed: int) -> tuple[str, float]:
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    cmd = [sys.executable, GENERATOR, "--corpus", str(per_pattern), "--seed", str(seed)]
    start = time.perf_counter()
    proc = subprocess.run(cmd, cwd=out_dir, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    seconds = time.perf_counter() - start
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr)
        raise SystemExit(f"The generator stopped with exit code {proc.returncode}.")
    return proc.stdout, seconds
```
- Exactly Day 4's idea: build the command `python make_questions.py --corpus 2000 --seed 42`, run it **inside the fresh folder** (`cwd=out_dir`, so the generator's relative file names work and its outputs land there), force UTF-8, time it, and if it fails print its output and stop. Returns the console output and the seconds. (This is why Day 3.1 reads files "from the current folder".)

### 4.5 `check_questions`

```python
def check_questions(raw: bytes, expected_types: int) -> tuple[list[list[str]], dict[str, object]]:
    reader = csv.reader(io.StringIO(raw.decode("utf-8-sig")))
    if next(reader) != QUESTION_HEADER:
        raise SystemExit("questions.csv has an unexpected header.")
    rows = [r for r in reader]
    texts = [r[0] for r in rows]
    problems: list[str] = []
    if len(set(texts)) != len(texts):
        problems.append("duplicate questions")
    if any(any(m in t for m in BROKEN_MARKS) for t in texts):
        problems.append("broken sentences")
    types = {r[2] for r in rows}
    if len(types) != expected_types:
        problems.append(f"{len(types)} question types instead of {expected_types}")
    if problems:
        raise SystemExit("Check FAILED: " + ", ".join(problems))
```
- A **simpler version of Day 4's `check_corpus`**. Parse the CSV from bytes (BOM removed by `utf-8-sig`), check the header with `next(reader)`, then three checks:
  1. **duplicates** (a set is shorter than the list);
  2. **broken sentences** (any `BROKEN_MARK` in any text; nested `any`);
  3. **number of question types** equals the expected 60.
- Any problem → stop with all problems listed. (There is no label check and no 4-column check here; Day 4 added those.)

```python
    lengths = [len(t) for t in texts]
    return rows, {
        "questions": len(rows),
        "patterns_with_questions": len({r[1] for r in rows}),
        "types": len(types),
        "characters": sum(lengths),
        "avg_chars": round(sum(lengths) / len(lengths), 2),
    }
```
- Returns the parsed rows (used later for examples) and basic statistics: number of questions, patterns that have questions, types, total characters, average length.

### 4.6 `write_zip`

```python
def write_zip(csv_path: Path, zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_path, arcname="questions.csv")
```
- Creates a zip (`"w"` = write; `ZIP_DEFLATED` = actually compress). `zf.write(file, arcname=...)` stores the file **under the name `questions.csv` inside the zip** (not the full path). Day 4's `read_corpus_bytes` looks for exactly one `questions.csv` inside a zip — this is the matching format.

### 4.7 `append_run_log`

```python
def append_run_log(log, machine, stats, per_pattern, seed, seconds, csv_sha, out_rel, zip_name) -> None:
    entry = [ "", f"## ... | {machine} | Day 3 question generation", "", ... "- Checks: PASS", "" ]
    with log.open("a", encoding="utf-8") as f:
        f.write("\n".join(entry))
```
- Same style as Day 4's log function: Markdown lines (machine, generator, settings, result counts, the **SHA-256 of `questions.csv`**, output folder and zip name, "Checks: PASS") appended to `run_log.md`. "PASS" is always written because a failed check stops the script earlier.

---

## 5. `parse_args` (same as Day 4's pattern)

```python
ap.add_argument("--machine", choices=["school", "home"], required=True)
ap.add_argument("--source-dir", type=Path, default=REPO_ROOT / "data/source")
ap.add_argument("--out-root", type=Path, default=REPO_ROOT / "runs/day3_generation")
ap.add_argument("--run-log", type=Path, default=REPO_ROOT / "run_log.md")
ap.add_argument("--per-pattern", type=int, default=2000, help="max questions per pattern (team setting: 2000)")
ap.add_argument("--seed", type=int, default=42, help="random seed (team setting: 42)")
ap.add_argument("--expected-types", type=int, default=60)
ap.add_argument("--examples", type=int, default=10, help="example questions to print")
ap.add_argument("--no-log", action="store_true")
```
- Same options as Day 4 (machine, source dir, work root, run log, per-pattern, seed, expected types, no-log). New: **`--examples`** (how many example questions to print, default 10). The report/team-corpus options of Day 4 do not exist here.

---

## 6. `main`

### 6.1 Setup and banner

```python
def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    a = parse_args()
    a.source_dir, a.out_root, a.run_log = a.source_dir.resolve(), a.out_root.resolve(), a.run_log.resolve()
```
- Same console safety as before. Here the three paths are made absolute in **one tuple-assignment line** (instead of Day 4's `setattr` loop).

```python
    started = datetime.now()
    stamp = started.strftime("%Y%m%d_%H%M%S")
    print("=" * 70)
    print(f" Day 3 question generation | machine={a.machine} | {started:%Y-%m-%d %H:%M:%S}")
    print(f" Python {platform.python_version()} | max {a.per_pattern} per pattern | seed {a.seed}")
    print("=" * 70)
```
- Remember the start time (`started`) and make a folder-safe stamp like `20261111_143012`.
- `"=" * 70` repeats the `=` character 70 times (a line). The banner shows machine, time, Python version, limit and seed — useful on the **video**.

### 6.2 Copy inputs and fingerprint them

```python
    src_files = check_inputs(a.source_dir)
    out_dir = a.out_root / stamp
    copy_inputs(a.source_dir, out_dir, src_files)
    print(f"Output folder: {out_dir}")
    print("Inputs (SHA-256, first 12 characters):")
    inputs = {p.name: sha256_bytes(p.read_bytes()) for p in [*(a.source_dir / n for n in REQUIRED_INPUTS), *src_files]}
    for name, h in inputs.items():
        print(f"  {name:<22} {h[:12]}")
```
- Check inputs, choose the time-stamped output folder, copy the files.
- `inputs` — a dictionary file name → fingerprint (same comprehension as Day 4's report inputs). Print each name (`<22` = left-aligned in 22 characters) and the **first 12 characters** of its fingerprint — proof on the video of which exact files were used.

### 6.3 Run the generator

```python
    print("\nRunning the team's generator: python make_questions.py "
          f"--corpus {a.per_pattern} --seed {a.seed}")
    generator_out, seconds = run_generator(out_dir, a.per_pattern, a.seed)
    print(generator_out.rstrip())
```
- Print the exact command, run it, and print what the generator printed (its totals table from Day 3.1). `rstrip()` removes trailing blank space.

### 6.4 Check, fingerprint and zip

```python
    csv_path = out_dir / "questions.csv"
    raw = csv_path.read_bytes()
    rows, stats = check_questions(raw, a.expected_types)
    csv_sha = sha256_bytes(raw)
    zip_name = f"questions_corpus_{a.machine}_{stamp}.zip"
    write_zip(csv_path, out_dir / zip_name)
```
- Read the generated `questions.csv` as bytes, run the checks, fingerprint it (`csv_sha`), name the zip `questions_corpus_<machine>_<time>.zip` and create it in the same folder.

### 6.5 The receipt `generation_info.json`

```python
    info = {
        "script": "scripts/day3_generate_questions.py",
        "script_version": SCRIPT_VERSION,
        "machine": a.machine,
        "started": started.isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "settings": {"per_pattern": a.per_pattern, "seed": a.seed},
        "inputs_sha256": inputs,
        "generation_seconds": round(seconds, 2),
        "stats": stats,
        "questions_csv_sha256": csv_sha,
        "zip": zip_name,
    }
    (out_dir / "generation_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
```
- Same kind of "receipt" as Day 5's manifest: script and version, machine, start time, Python version, the **settings**, the fingerprints of the **inputs** and of the **output**, time, statistics and the zip name. Saved as readable JSON (Japanese kept).

### 6.6 Summary and examples

```python
    print(f"\nGeneration time: {seconds:.1f} s")
    print(f"Questions: {stats['questions']:,} | types: {stats['types']} | "
          f"patterns with questions: {stats['patterns_with_questions']} | avg chars: {stats['avg_chars']}")
    print("Check: PASS (no duplicates, no unfilled slots, 60 question types)")
    print(f"questions.csv SHA-256: {csv_sha}")
    print(f"Zip for the other PC: {out_dir / zip_name}")
```
- Print the time, the counts, the PASS message (reached only if the checks passed), the **full fingerprint** of `questions.csv`, and where the zip is. (The text "60 question types" is fixed text; the real check used `--expected-types`.)

```python
    print(f"\n{a.examples} example questions (random, seed {a.seed}):")
    for text, pid, tid, _ in random.Random(a.seed).sample(rows, min(a.examples, len(rows))):
        print(f"  [{tid} {pid}] {text}")
```
- `random.Random(a.seed).sample(rows, k)` picks `k` **different** random rows; seeded, so the same examples every time. `min(a.examples, len(rows))` avoids asking for more than exist.
- Each row is unpacked into its four columns (`_` ignores the meaning label) and printed as `[T05 P012] question text` — nice for the video.

### 6.7 Log and exit

```python
    if not a.no_log:
        out_rel = str(out_dir.relative_to(REPO_ROOT)).replace("\\", "/")
        append_run_log(a.run_log, a.machine, stats, a.per_pattern, a.seed, seconds, csv_sha, out_rel, zip_name)
        print(f"\nAppended an entry to {a.run_log.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```
- Unless `--no-log`: write the path relative to the repo with `/` (same trick as Day 4/5) and append to `run_log.md`. Return 0. Same ending as the other scripts.

---

## 7. The whole run in order

1. Options → absolute paths → banner.
2. `check_inputs` → new time-stamped folder → `copy_inputs` → fingerprints of inputs.
3. `run_generator` (with `cwd` = the fresh folder).
4. `check_questions` → fingerprint → zip.
5. `generation_info.json` → summary → 10 random examples → run log.

**What is in the output folder afterwards:** copies of the inputs, the generator's files (`patterns.csv`, `question_types.csv`, `columns.csv`, `samples.csv`, `questions.csv`), `questions_corpus_<machine>_<time>.zip` and `generation_info.json`.

## 8. How to read the results

- **PASS** = no duplicates, no unfilled slots/broken joins, exactly the expected number of question types.
- **`questions.csv SHA-256`** — the fingerprint of the corpus. If another PC (same inputs, same seed, same limit) produces the **same** SHA-256, the generation is reproducible (this is what Day 4 compares).
- **Inputs' SHA-256 (first 12)** — proves which version of the team's files was used.
- **Example questions** — a quick human check that sentences look natural.

## 9. Small things worth knowing

- **Do not commit `data/source/` or the generated files to the public school repository** (docstring warning).
- **Weaker check than Day 4.** `check_questions` does not verify that every row has 4 columns or that each label matches its pattern/type. The generator never produces such rows, but if one existed, the example printing (which unpacks 4 columns) would crash with a Python error.
- **"PASS (…60 question types)" is fixed text.** The real expected number comes from `--expected-types`.
- **`relative_to(REPO_ROOT)`** raises an error if you point `--out-root` outside the repo (same caveat as Day 4/5).
- **Same inputs + same seed + same limit = same output.** Different machines should give the same `questions.csv` SHA-256 (the generator uses seeded randomness and sorted lists; see Day 3.1).

## 10. Quick summary in one paragraph

The script copies the team's inputs into a fresh time-stamped folder, fingerprints them, runs the team's generator there with fixed settings (2,000 per pattern, seed 42), checks the result for duplicates, broken sentences and the right number of question types, saves a zip to carry to the other PC and a JSON receipt with all fingerprints, prints a summary with 10 random example questions, and appends a note to `run_log.md`.
