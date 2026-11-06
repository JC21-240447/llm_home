r"""Day 4 - generate the question corpus with the team's generator, check it, and report.
Day 4 - チームの生成プログラムで質問コーパスを作り、検査し、報告する

==========================  ENGLISH  ==========================

The generator itself is the team's `data/source/make_questions.py`. It fills
the team's sentence patterns (src_*.txt) with words from the team's glossary,
synonym dictionary and 50-row sample table (DB.xlsx).

This script runs that generator the safe and repeatable way:
1. Checks that every input file exists and that pandas/openpyxl are installed.
2. Copies the inputs into a fresh work folder (runs/day4_generation/<time>/),
   because the generator writes its outputs into the folder it runs in and
   would otherwise overwrite the team's files in data/source.
3. Runs the generator there with a fixed seed and a per-pattern limit.
4. Checks the result: header, no duplicates, no broken sentences, readable
   labels, 60 question types, sentence lengths.
5. Compares the result with a reference corpus: the team's
   data/source/questions_corpus.zip by default, or the zip made by Day 3 on
   another PC (--reference path\to\zip), and with the team's patterns.csv / question_types.csv / columns.csv:
   byte-identical or not, and which patterns differ.
6. Writes docs/DAY4_GENERATION_REPORT.md and a run_log.md entry.

Usage (VS Code terminal, repo root, venv python):
    .venv\Scripts\python.exe scripts\day4_generate_corpus.py --machine home

SCHOOL-PC SWAP NOTE:
    CPU only; the code is the same. On the school PC run it with
    --machine school. pandas and openpyxl must be installed there too
    (pip install -r requirements.txt inside that PC's .venv).

==========================  日本語  ==========================

生成プログラムそのものは、チームの `data/source/make_questions.py` です。
チームの文型(src_*.txt)に、チームの用語集・同義語辞書・50行のサンプル表(DB.xlsx)
の言葉を当てはめて質問文を作ります。

このスクリプトは、その生成プログラムを安全に、何度でも同じように実行します:
1. 入力ファイルがすべてあること、pandas と openpyxl が入っていることを確認する。
2. 入力を新しい作業フォルダー(runs/day4_generation/<時刻>/)にコピーする。
   生成プログラムは、実行したフォルダーに出力を書くので、そのままだと
   data/source にあるチームのファイルを上書きしてしまうため。
3. 固定したシードと、文型ごとの上限を指定して、そこで生成プログラムを実行する。
4. 結果を検査する: ヘッダー、重複なし、壊れた文なし、ラベルが読める、
   質問タイプが60、文の長さ。
5. 結果を基準のコーパス(既定はチームの data/source/questions_corpus.zip。
   --reference で別のPCの Day 3 の zip も指定できる)や、チームの
   patterns.csv / question_types.csv / columns.csv と比べる:
   1バイトも違わないか、違うならどの文型か。
6. docs/DAY4_GENERATION_REPORT.md を書き、run_log.md に追記する。

使い方(VS Codeターミナル、リポジトリのルートで、venv の python):
    .venv\Scripts\python.exe scripts\day4_generate_corpus.py --machine home

学校PCで使う場合の注意:
    CPU だけを使い、コードは同じです。学校PCでは --machine school を付けて実行します。
    学校PCの .venv にも pandas と openpyxl が必要です
    (その PC の .venv で pip install -r requirements.txt)。
"""
from __future__ import annotations

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

SCRIPT_VERSION: str = "day4-v2"
REPO_ROOT: Path = Path(__file__).resolve().parent.parent
GENERATOR: str = "make_questions.py"
REQUIRED_INPUTS: tuple[str, ...] = (GENERATOR, "用語集.csv", "同義語辞書.csv", "DB.xlsx")
SUMMARY_FILES: tuple[str, ...] = ("patterns.csv", "question_types.csv", "columns.csv")
QUESTION_HEADER: list[str] = ["質問", "パターンID", "質問タイプID", "意味"]
BROKEN_MARKS: tuple[str, ...] = ("{", "}", "のの", "社員の社員", "平均平均")


@dataclass
class GeneratedCorpus:
    """The generator's questions.csv, parsed."""

    raw_bytes: bytes
    rows: list[list[str]]  # 質問, パターンID, 質問タイプID, 意味


# ================================================================ helpers
def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def check_inputs(source_dir: Path) -> list[Path]:
    """Stop with a clear message if an input file or library is missing. Returns the src_*.txt files."""
    missing = [n for n in REQUIRED_INPUTS if not (source_dir / n).exists()]
    src_files = sorted(source_dir.glob("src_*.txt"))
    if not src_files:
        missing.append("src_*.txt")
    if missing:
        raise SystemExit(
            f"Missing in {source_dir}: {', '.join(missing)}\n"
            "Put the team's files there first (see Step 1 of the Day 4 walkthrough)."
        )
    libs = [m for m in ("pandas", "openpyxl") if importlib.util.find_spec(m) is None]
    if libs:
        raise SystemExit(
            f"Missing Python libraries: {', '.join(libs)}\n"
            r"Install them: .venv\Scripts\python.exe -m pip install -r requirements.txt"
        )
    return src_files


def prepare_work_dir(source_dir: Path, work_root: Path, src_files: list[Path]) -> Path:
    """Copy the generator and its inputs into a fresh folder, so data/source is never written to."""
    work = work_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    work.mkdir(parents=True, exist_ok=False)
    for name in REQUIRED_INPUTS:
        shutil.copy2(source_dir / name, work / name)
    for f in src_files:
        shutil.copy2(f, work / f.name)
    return work


def run_generator(work: Path, per_pattern: int, seed: int) -> tuple[str, float]:
    """Run the team's generator in the work folder with this venv's Python. Returns (console output, seconds)."""
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    cmd = [sys.executable, GENERATOR, "--corpus", str(per_pattern), "--seed", str(seed)]
    start = time.perf_counter()
    proc = subprocess.run(cmd, cwd=work, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    seconds = time.perf_counter() - start
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr)
        raise SystemExit(f"The generator stopped with exit code {proc.returncode}. See the error above.")
    return proc.stdout, seconds


def read_corpus_bytes(path: Path) -> bytes:
    """questions.csv bytes, from a .csv or from the questions.csv inside a .zip."""
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            members = [n for n in zf.namelist() if n.endswith("questions.csv")]
            if len(members) != 1:
                raise SystemExit(f"{path}: expected one questions.csv inside, found {members}")
            return zf.read(members[0])
    return path.read_bytes()


def parse_corpus(raw: bytes) -> GeneratedCorpus:
    reader = csv.reader(io.StringIO(raw.decode("utf-8-sig")))
    header = next(reader)
    if header != QUESTION_HEADER:
        raise SystemExit(f"Unexpected header {header}")
    return GeneratedCorpus(raw, [row for row in reader])


# ================================================================ checks
def check_corpus(corpus: GeneratedCorpus, expected_types: int) -> dict[str, object]:
    """Quality checks on the generated questions. Raises SystemExit if any check fails."""
    texts = [r[0] for r in corpus.rows]
    problems: list[str] = []
    if any(len(r) != 4 for r in corpus.rows):
        problems.append("a row does not have 4 columns")
    if len(set(texts)) != len(texts):
        problems.append(f"{len(texts) - len(set(texts))} duplicate questions")
    broken = [t for t in texts if any(m in t for m in BROKEN_MARKS)]
    if broken:
        problems.append(f"{len(broken)} broken sentences, e.g. {broken[0]}")
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
    types = {r[2] for r in corpus.rows}
    if len(types) != expected_types:
        problems.append(f"{len(types)} question types, expected {expected_types}")
    if problems:
        raise SystemExit("Corpus check FAILED: " + "; ".join(problems))
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


def compare_with_team(new: GeneratedCorpus, team_raw: bytes | None) -> dict[str, object]:
    """Byte comparison plus a per-pattern comparison of the question sets."""
    if team_raw is None:
        return {"available": False}
    team = parse_corpus(team_raw)

    def by_pattern(c: GeneratedCorpus) -> dict[str, set[str]]:
        d: dict[str, set[str]] = defaultdict(set)
        for text, pid, _, _ in c.rows:
            d[pid].add(text)
        return d

    a, b = by_pattern(new), by_pattern(team)
    all_pids = sorted(set(a) | set(b))
    differing = [p for p in all_pids if a.get(p) != b.get(p)]
    return {
        "available": True,
        "byte_identical": new.raw_bytes == team_raw,
        "team_questions": len(team.rows),
        "patterns_identical": len(all_pids) - len(differing),
        "patterns_differing": differing,
    }


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


# ================================================================ report
def load_pattern_table(work: Path) -> list[dict[str, str]]:
    with (work / "patterns.csv").open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_report(
    path: Path,
    stats: dict[str, object],
    cmp_team: dict[str, object],
    cmp_files: dict[str, str],
    patterns: list[dict[str, str]],
    corpus: GeneratedCorpus,
    inputs: dict[str, str],
    per_pattern: int,
    seed: int,
    seconds: float,
    generator_out: str,
    reference_name: str,
) -> None:
    """docs/DAY4_GENERATION_REPORT.md. Shows questions only, not the internal labels."""
    gen_count = Counter(r[1] for r in corpus.rows)
    type_rows: dict[str, list] = {}
    for p in patterns:
        t = type_rows.setdefault(p["質問タイプID"], [p["質問タイプ"], 0, 0, 0])
        t[1] += 1
        t[2] += int(p["作れる質問の数"])
        t[3] += gen_count.get(p["パターンID"], 0)
    first_question: dict[str, str] = {}
    for text, _, tid, _ in corpus.rows:
        first_question.setdefault(tid, text)
    styles = Counter(p["文体"] for p in patterns)
    empty = [p["パターンID"] for p in patterns if gen_count.get(p["パターンID"], 0) == 0]
    possible = sum(int(p["作れる質問の数"]) for p in patterns)

    lines = [
        "# Day 4 report: question generation",
        "",
        f"Generated by `scripts/day4_generate_corpus.py` ({SCRIPT_VERSION}) on {datetime.now():%Y-%m-%d %H:%M}.",
        "",
        "## Inputs (all made by the team)",
        "",
        "| File | SHA-256 |",
        "| --- | --- |",
        *(f"| `{n}` | `{h}` |" for n, h in inputs.items()),
        "",
        "## Settings",
        "",
        f"- At most {per_pattern:,} questions per pattern (balancing), seed {seed}.",
        f"- Generation time: {seconds:.1f} seconds.",
        "",
        "## Result",
        "",
        "| Item | Value |",
        "| --- | --- |",
        f"| Patterns | {len(patterns)} ({', '.join(f'{k} {v}' for k, v in styles.items())}) |",
        f"| Question types | {stats['types']} |",
        f"| All distinct questions the patterns can make | {possible:,} |",
        f"| Questions written (after the per-pattern limit and duplicate removal) | {stats['questions']:,} |",
        f"| Patterns with at least one question | {stats['patterns_with_questions']} |",
        f"| Patterns with no question left after duplicate removal | {', '.join(empty) or 'none'} |",
        f"| Characters | {stats['characters']:,} |",
        f"| Length per question (min / average / max) | {stats['min_chars']} / {stats['avg_chars']} / {stats['max_chars']} |",
        "",
        "Checks passed: no duplicates, no unfilled slots or broken joins, every row has a readable label",
        "that matches its pattern and type.",
        "",
        "## Reproducibility",
        "",
    ]
    if cmp_team.get("available"):
        lines += [
            f"- Compared with the reference corpus (`{reference_name}`, {cmp_team['team_questions']:,} questions):",
            f"  byte-identical: **{'yes' if cmp_team['byte_identical'] else 'no'}**;"
            f" identical patterns: {cmp_team['patterns_identical']};"
            f" differing patterns: {', '.join(cmp_team['patterns_differing']) or 'none'}.",
        ]
    else:
        lines.append("- The reference corpus was not found, so no comparison was made.")
    lines += [f"- `{n}` compared with the team's copy: {v}." for n, v in cmp_files.items()]
    lines += [
        "",
        "## Per question type (one example each)",
        "",
        "| Type | Name | Patterns | Possible questions | Written | Example |",
        "| --- | --- | --- | --- | --- | --- |",
        *(
            f"| {tid} | {v[0]} | {v[1]} | {v[2]:,} | {v[3]:,} | {first_question.get(tid, '')} |"
            for tid, v in sorted(type_rows.items())
        ),
        "",
        "## Generator console output",
        "",
        "```text",
        generator_out.strip(),
        "```",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def append_run_log(log: Path, machine: str, stats: dict[str, object], cmp_team: dict[str, object],
                   per_pattern: int, seed: int, seconds: float, work_rel: str, reference_name: str) -> None:
    same = "n/a" if not cmp_team.get("available") else ("yes" if cmp_team["byte_identical"] else
                                                        f"no ({len(cmp_team['patterns_differing'])} patterns differ)")
    entry = [
        "",
        f"## {datetime.now():%Y-%m-%d %H:%M} | {machine} | Day 4 question generation",
        "",
        f"- Machine label: {machine} (CPU only; no GPU used)",
        f"- Generator: data/source/make_questions.py (team), run by scripts/day4_generate_corpus.py ({SCRIPT_VERSION})",
        f"- Settings: max {per_pattern} per pattern, seed {seed}; time {seconds:.1f} s",
        f"- Result: {stats['questions']:,} questions, {stats['types']} types, "
        f"{stats['patterns_with_questions']} patterns with questions, avg {stats['avg_chars']} chars",
        f"- Identical to the reference corpus ({reference_name}): {same}",
        f"- Output folder: {work_rel}",
        "- Checks: PASS",
        "",
    ]
    with log.open("a", encoding="utf-8") as f:
        f.write("\n".join(entry))


# ================================================================== main
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Day 4: run the team's question generator, check and report.")
    ap.add_argument("--machine", choices=["school", "home"], required=True)
    ap.add_argument("--source-dir", type=Path, default=REPO_ROOT / "data/source")
    ap.add_argument("--work-root", type=Path, default=REPO_ROOT / "runs/day4_generation")
    ap.add_argument("--team-corpus", "--reference", dest="team_corpus", type=Path,
                    default=REPO_ROOT / "data/source/questions_corpus.zip",
                    help="corpus to compare with: the team's zip (default) or a Day 3 zip from another PC")
    ap.add_argument("--report", type=Path, default=REPO_ROOT / "docs/DAY4_GENERATION_REPORT.md")
    ap.add_argument("--run-log", type=Path, default=REPO_ROOT / "run_log.md")
    ap.add_argument("--per-pattern", type=int, default=2000, help="max questions per pattern (team setting: 2000)")
    ap.add_argument("--seed", type=int, default=42, help="random seed (team setting: 42)")
    ap.add_argument("--expected-types", type=int, default=60)
    ap.add_argument("--no-log", action="store_true")
    return ap.parse_args()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    a = parse_args()
    for name in ("source_dir", "work_root", "team_corpus", "report", "run_log"):
        setattr(a, name, getattr(a, name).resolve())

    print(f"=== Day 4 question generation | machine={a.machine} | max {a.per_pattern}/pattern | seed {a.seed} ===")
    src_files = check_inputs(a.source_dir)
    work = prepare_work_dir(a.source_dir, a.work_root, src_files)
    print(f"Work folder: {work}")

    generator_out, seconds = run_generator(work, a.per_pattern, a.seed)
    print(generator_out.strip())
    print(f"Generation took {seconds:.1f} s")

    corpus = parse_corpus((work / "questions.csv").read_bytes())
    stats = check_corpus(corpus, a.expected_types)
    team_raw = read_corpus_bytes(a.team_corpus) if a.team_corpus.exists() else None
    cmp_team = compare_with_team(corpus, team_raw)
    cmp_files = compare_summary_files(work, a.source_dir)

    inputs = {f.name: sha256_file(f) for f in [*(a.source_dir / n for n in REQUIRED_INPUTS), *src_files]}
    a.report.parent.mkdir(parents=True, exist_ok=True)
    write_report(a.report, stats, cmp_team, cmp_files, load_pattern_table(work), corpus, inputs,
                 a.per_pattern, a.seed, seconds, generator_out, a.team_corpus.name)
    (work / "check_result.json").write_text(
        json.dumps({"stats": stats, "compare_team": cmp_team, "compare_files": cmp_files,
                    "questions_csv_sha256": sha256_bytes(corpus.raw_bytes)}, ensure_ascii=False, indent=2),
        encoding="utf-8")

    print()
    print(f"Questions: {stats['questions']:,} | types: {stats['types']} | "
          f"patterns with questions: {stats['patterns_with_questions']} | avg chars: {stats['avg_chars']}")
    print("Corpus check: PASS (no duplicates, no broken sentences, labels readable)")
    if cmp_team.get("available"):
        print(f"Same as the reference corpus ({a.team_corpus.name}): byte-identical={cmp_team['byte_identical']}, "
              f"identical patterns={cmp_team['patterns_identical']}, "
              f"differing patterns={cmp_team['patterns_differing'] or 'none'}")
    for n, v in cmp_files.items():
        print(f"{n}: {v}")
    print(f"Report: {a.report}")
    if not a.no_log:
        work_rel = str(work.relative_to(REPO_ROOT)).replace("\\", "/")
        append_run_log(a.run_log, a.machine, stats, cmp_team, a.per_pattern, a.seed, seconds, work_rel,
                       a.team_corpus.name)
        print(f"Appended an entry to {a.run_log.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
