r"""Day 3 - generate the question corpus with the team's generator (the run shown on video).
Day 3 - チームの生成プログラムで質問コーパスを作る(動画で記録する実行)

==========================  ENGLISH  ==========================

Runs the team's generator data/source/make_questions.py with the team's
settings (at most 2,000 questions per pattern, seed 42) and keeps the result:

1. Checks that every input file exists and that pandas/openpyxl are installed.
2. Copies the inputs into a new output folder runs/day3_generation/<time>/,
   because the generator writes its files into the folder it runs in.
3. Runs the generator and times it.
4. Checks the result (no duplicates, no unfilled slots, 60 question types).
5. Writes, in the same folder:
     questions.csv                         the generated questions
     questions_corpus_<machine>_<time>.zip  the same file zipped, to carry to the other PC
     generation_info.json                  settings, input and output SHA-256, counts, time
6. Prints a summary and 10 example questions, and appends to run_log.md.

Day 4 later regenerates on another machine and compares with this zip.

Usage (VS Code terminal, repo root, venv python):
    .venv\Scripts\python.exe scripts\day3_generate_questions.py --machine home

SCHOOL-PC SWAP NOTE:
    Same code. On the school PC use --machine school. Install pandas and
    openpyxl into that PC's .venv only (never into the school's Python):
        .venv\Scripts\python.exe -m pip install pandas openpyxl
    The school repository is public: do not commit data/source/ or the
    generated files there (they contain the team's table and internal labels).

==========================  日本語  ==========================

チームの生成プログラム data/source/make_questions.py を、チームと同じ設定
(1文型につき最大2,000問、シード42)で実行し、結果を保存します:

1. 入力ファイルがすべてあること、pandas と openpyxl が入っていることを確認する。
2. 入力を新しい出力フォルダー runs/day3_generation/<時刻>/ にコピーする
   (生成プログラムは、実行したフォルダーにファイルを書くため)。
3. 生成プログラムを実行し、時間を測る。
4. 結果を検査する(重複なし、埋まっていないスロットなし、質問タイプが60)。
5. 同じフォルダーに次を書き出す:
     questions.csv                         生成した質問
     questions_corpus_<machine>_<時刻>.zip  同じファイルの zip(別のPCに持っていく用)
     generation_info.json                  設定、入力と出力の SHA-256、件数、時間
6. まとめと質問の例10個を表示し、run_log.md に追記する。

Day 4 では、別のマシンで作り直して、この zip と比べます。

使い方(VS Codeターミナル、リポジトリのルートで、venv の python):
    .venv\Scripts\python.exe scripts\day3_generate_questions.py --machine home

学校PCで使う場合の注意:
    コードは同じです。学校PCでは --machine school を使います。pandas と openpyxl は、
    その PC の .venv の中だけに入れます(学校の Python には入れない):
        .venv\Scripts\python.exe -m pip install pandas openpyxl
    学校のリポジトリは公開されているので、data/source/ や生成したファイルは
    コミットしないでください(チームの表と内部用のラベルが入っているため)。
"""
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

SCRIPT_VERSION: str = "day3-v1"
REPO_ROOT: Path = Path(__file__).resolve().parent.parent
GENERATOR: str = "make_questions.py"
REQUIRED_INPUTS: tuple[str, ...] = (GENERATOR, "用語集.csv", "同義語辞書.csv", "DB.xlsx")
QUESTION_HEADER: list[str] = ["質問", "パターンID", "質問タイプID", "意味"]
BROKEN_MARKS: tuple[str, ...] = ("{", "}", "のの", "社員の社員", "平均平均")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def check_inputs(source_dir: Path) -> list[Path]:
    """Stop with a clear message if an input or a library is missing. Returns the src_*.txt files."""
    missing = [n for n in REQUIRED_INPUTS if not (source_dir / n).exists()]
    src_files = sorted(source_dir.glob("src_*.txt"))
    if not src_files:
        missing.append("src_*.txt")
    if missing:
        raise SystemExit(f"Missing in {source_dir}: {', '.join(missing)}")
    libs = [m for m in ("pandas", "openpyxl") if importlib.util.find_spec(m) is None]
    if libs:
        raise SystemExit(
            f"Missing libraries: {', '.join(libs)}. Install them into this .venv only:\n"
            r"  .venv\Scripts\python.exe -m pip install pandas openpyxl"
        )
    return src_files


def copy_inputs(source_dir: Path, out_dir: Path, src_files: list[Path]) -> None:
    """Copy the generator and its inputs, so the team's folder data/source is never written to."""
    out_dir.mkdir(parents=True, exist_ok=False)
    for name in REQUIRED_INPUTS:
        shutil.copy2(source_dir / name, out_dir / name)
    for f in src_files:
        shutil.copy2(f, out_dir / f.name)


def run_generator(out_dir: Path, per_pattern: int, seed: int) -> tuple[str, float]:
    """Run make_questions.py with this venv's Python. Returns (console output, seconds)."""
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    cmd = [sys.executable, GENERATOR, "--corpus", str(per_pattern), "--seed", str(seed)]
    start = time.perf_counter()
    proc = subprocess.run(cmd, cwd=out_dir, env=env, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    seconds = time.perf_counter() - start
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr)
        raise SystemExit(f"The generator stopped with exit code {proc.returncode}.")
    return proc.stdout, seconds


def check_questions(raw: bytes, expected_types: int) -> tuple[list[list[str]], dict[str, object]]:
    """Parse questions.csv and run the quality checks. Raises SystemExit on any failure."""
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
    lengths = [len(t) for t in texts]
    return rows, {
        "questions": len(rows),
        "patterns_with_questions": len({r[1] for r in rows}),
        "types": len(types),
        "characters": sum(lengths),
        "avg_chars": round(sum(lengths) / len(lengths), 2),
    }


def write_zip(csv_path: Path, zip_path: Path) -> None:
    """Zip questions.csv (as 'questions.csv' inside) for carrying to the other machine."""
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_path, arcname="questions.csv")


def append_run_log(log: Path, machine: str, stats: dict[str, object], per_pattern: int, seed: int,
                   seconds: float, csv_sha: str, out_rel: str, zip_name: str) -> None:
    entry = [
        "",
        f"## {datetime.now():%Y-%m-%d %H:%M} | {machine} | Day 3 question generation",
        "",
        f"- Machine label: {machine} (CPU only; no GPU used)",
        f"- Generator: data/source/make_questions.py (team), run by scripts/day3_generate_questions.py ({SCRIPT_VERSION})",
        f"- Settings: max {per_pattern} per pattern, seed {seed}; generation time {seconds:.1f} s",
        f"- Result: {stats['questions']:,} questions, {stats['types']} types, "
        f"{stats['patterns_with_questions']} patterns with questions, avg {stats['avg_chars']} chars",
        f"- questions.csv SHA-256: {csv_sha}",
        f"- Output: {out_rel}/ (zip for the other PC: {zip_name})",
        "- Checks: PASS",
        "",
    ]
    with log.open("a", encoding="utf-8") as f:
        f.write("\n".join(entry))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Day 3: generate the question corpus with the team's generator.")
    ap.add_argument("--machine", choices=["school", "home"], required=True)
    ap.add_argument("--source-dir", type=Path, default=REPO_ROOT / "data/source")
    ap.add_argument("--out-root", type=Path, default=REPO_ROOT / "runs/day3_generation")
    ap.add_argument("--run-log", type=Path, default=REPO_ROOT / "run_log.md")
    ap.add_argument("--per-pattern", type=int, default=2000, help="max questions per pattern (team setting: 2000)")
    ap.add_argument("--seed", type=int, default=42, help="random seed (team setting: 42)")
    ap.add_argument("--expected-types", type=int, default=60)
    ap.add_argument("--examples", type=int, default=10, help="example questions to print")
    ap.add_argument("--no-log", action="store_true")
    return ap.parse_args()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    a = parse_args()
    a.source_dir, a.out_root, a.run_log = a.source_dir.resolve(), a.out_root.resolve(), a.run_log.resolve()

    started = datetime.now()
    stamp = started.strftime("%Y%m%d_%H%M%S")
    print("=" * 70)
    print(f" Day 3 question generation | machine={a.machine} | {started:%Y-%m-%d %H:%M:%S}")
    print(f" Python {platform.python_version()} | max {a.per_pattern} per pattern | seed {a.seed}")
    print("=" * 70)

    src_files = check_inputs(a.source_dir)
    out_dir = a.out_root / stamp
    copy_inputs(a.source_dir, out_dir, src_files)
    print(f"Output folder: {out_dir}")
    print("Inputs (SHA-256, first 12 characters):")
    inputs = {p.name: sha256_bytes(p.read_bytes()) for p in [*(a.source_dir / n for n in REQUIRED_INPUTS), *src_files]}
    for name, h in inputs.items():
        print(f"  {name:<22} {h[:12]}")

    print("\nRunning the team's generator: python make_questions.py "
          f"--corpus {a.per_pattern} --seed {a.seed}")
    generator_out, seconds = run_generator(out_dir, a.per_pattern, a.seed)
    print(generator_out.rstrip())

    csv_path = out_dir / "questions.csv"
    raw = csv_path.read_bytes()
    rows, stats = check_questions(raw, a.expected_types)
    csv_sha = sha256_bytes(raw)
    zip_name = f"questions_corpus_{a.machine}_{stamp}.zip"
    write_zip(csv_path, out_dir / zip_name)

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

    print(f"\nGeneration time: {seconds:.1f} s")
    print(f"Questions: {stats['questions']:,} | types: {stats['types']} | "
          f"patterns with questions: {stats['patterns_with_questions']} | avg chars: {stats['avg_chars']}")
    print("Check: PASS (no duplicates, no unfilled slots, 60 question types)")
    print(f"questions.csv SHA-256: {csv_sha}")
    print(f"Zip for the other PC: {out_dir / zip_name}")

    print(f"\n{a.examples} example questions (random, seed {a.seed}):")
    for text, pid, tid, _ in random.Random(a.seed).sample(rows, min(a.examples, len(rows))):
        print(f"  [{tid} {pid}] {text}")

    if not a.no_log:
        out_rel = str(out_dir.relative_to(REPO_ROOT)).replace("\\", "/")
        append_run_log(a.run_log, a.machine, stats, a.per_pattern, a.seed, seconds, csv_sha, out_rel, zip_name)
        print(f"\nAppended an entry to {a.run_log.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
