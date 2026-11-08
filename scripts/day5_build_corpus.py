r"""Day 5 - corpus split, light balancing, statistics and data card (Model A).
Day 5 - コーパスの分割、軽いバランス調整、統計、データカード(Model A)

==========================  ENGLISH  ==========================

Input (made by the team, treated as read-only source of truth):
    data/source/questions_corpus.zip   (contains questions.csv:
                                        質問, パターンID, 質問タイプID, 意味)
    data/source/patterns.csv           (pattern -> question type, style)
    data/source/用語集.csv              (optional, for the coverage report)
    data/source/同義語辞書.csv           (optional, for the coverage report)

What it does
------------
1. Loads and validates the questions (header, no empty or multi-line text,
   no duplicates, one type per pattern, readable 意味 labels).
2. Splits BY PATTERN, never by line: whole patterns (= whole phrasings) are
   held out for validation and test, one pattern per chosen question type,
   so the model is evaluated on wording it has never seen.
3. Removes near-duplicates across splits (same text after ignoring
   punctuation and spaces) from validation/test.
4. Caps each held-out pattern so one huge pattern cannot dominate evaluation.
5. Oversamples small question types in TRAIN only, a few times at most.
6. Writes train/val/test text files (question text only), internal label
   files, a split manifest with SHA-256 hashes, docs/DATA_CARD.md, and a
   run_log.md entry.

Only the 質問 text is used for training. The 意味 labels are written to
separate *_labels.csv files and are never mixed into the training text.

Usage (VS Code terminal, repo root, venv python, no activation needed):
    .venv\Scripts\python.exe scripts\day5_build_corpus.py --machine home

SCHOOL-PC SWAP NOTE:
    This script uses only the CPU and the standard library. The code is the
    same on the school PC; only run it with --machine school there.
    The split is fully determined by --seed and the input files, so both
    machines produce byte-identical outputs (check the SHA-256 values in
    data/splits/split_manifest.json).

==========================  日本語  ==========================

入力(チームが作成。変更しない正本として扱う):
    data/source/questions_corpus.zip   (中に questions.csv:
                                        質問, パターンID, 質問タイプID, 意味)
    data/source/patterns.csv           (パターン → 質問タイプ、文体)
    data/source/用語集.csv              (任意。カバー率の報告に使う)
    data/source/同義語辞書.csv           (任意。カバー率の報告に使う)

このスクリプトが行うこと
------------------------
1. 質問を読み込み、検査する(ヘッダー、空や複数行の文が無いこと、重複が無いこと、
   1つのパターンは1つの質問タイプに属すること、意味ラベルが読めること)。
2. 行ではなく「パターン単位」で分割する。選んだ質問タイプごとに1つのパターン
   (= 1つの言い回し)を丸ごと検証用・テスト用に取り分けるので、モデルは
   一度も見たことのない言い回しで評価される。
3. 句読点と空白を無視すると同じになる文(ほぼ重複)を、検証用・テスト用から除く。
4. 取り分けたパターンごとに上限を設け、1つの巨大なパターンが評価を支配しないようにする。
5. 小さな質問タイプを、学習用(train)の中だけで数回まで増やす(オーバーサンプリング)。
6. train/val/test のテキスト(質問文のみ)、内部用のラベルファイル、SHA-256 付きの
   分割マニフェスト、docs/DATA_CARD.md を書き出し、run_log.md に追記する。

学習に使うのは 質問 の文だけです。意味 のラベルは別の *_labels.csv に書き出し、
学習用テキストには混ぜません。

使い方(VS Codeターミナル、リポジトリのルートで。venv の python を直接指定、有効化は不要):
    .venv\Scripts\python.exe scripts\day5_build_corpus.py --machine home

学校PCで使う場合の注意:
    このスクリプトは CPU と標準ライブラリだけを使います。学校PCでもコードは同じで、
    --machine school を付けて実行するだけです。分割は --seed と入力ファイルだけで
    決まるので、両方のマシンでまったく同じ出力になります
    (data/splits/split_manifest.json の SHA-256 で確認できます)。
"""
from __future__ import annotations

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

SCRIPT_VERSION: str = "day5-v1"
REPO_ROOT: Path = Path(__file__).resolve().parent.parent
SPLITS: tuple[str, str, str] = ("train", "val", "test")
QUESTION_HEADER: list[str] = ["質問", "パターンID", "質問タイプID", "意味"]


# ============================================================== data classes
@dataclass(frozen=True)
class Question:
    """One generated question and its internal labels."""

    text: str
    pattern_id: str
    type_id: str
    meaning: str  # JSON string, kept aside, never used as training text


@dataclass(frozen=True)
class PatternInfo:
    """One row of patterns.csv."""

    pattern_id: str
    type_id: str
    type_name: str
    style: str
    template: str


@dataclass
class SplitConfig:
    """All knobs of the split. Every value is written into the manifest."""

    seed: int = 42
    val_fraction: float = 0.07       # share of non-empty patterns held out for validation
    test_fraction: float = 0.07      # share of non-empty patterns held out for test
    min_type_lines: int = 100        # a type needs this many questions to donate a pattern
    min_pattern_lines: int = 20      # a held-out pattern needs this many questions
    max_donate_share: float = 0.4    # ...and may hold at most this share of its type's questions
    eval_cap_per_pattern: int = 500  # at most this many questions per held-out pattern
    oversample_target: int = 200     # small types are repeated up to about this size...
    max_repeat: int = 5              # ...but never more than this many copies


@dataclass
class SplitResult:
    """Everything produced by the split, before writing files."""

    assignment: dict[str, str]                     # pattern_id -> split
    lines: dict[str, list[Question]]               # split -> questions (train after oversampling)
    train_unique: list[Question]                   # train before oversampling
    repeat_factor: dict[str, int]                  # type_id -> copies in train
    dropped_near_dup: dict[str, int] = field(default_factory=dict)
    dropped_by_cap: dict[str, int] = field(default_factory=dict)


# ================================================================== loading
def sha256_of(path: Path) -> str:
    """SHA-256 of a file, so anyone can prove the inputs and outputs match."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_questions_text(source: Path) -> str:
    """Return the text of questions.csv, read from a .zip or a plain .csv."""
    if source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as zf:
            members = [n for n in zf.namelist() if n.endswith("questions.csv")]
            if len(members) != 1:
                raise SystemExit(f"{source}: expected exactly one questions.csv inside, found {members}")
            return zf.read(members[0]).decode("utf-8-sig")
    return source.read_text(encoding="utf-8-sig")


def load_questions(source: Path) -> list[Question]:
    """Load and validate the question corpus. Stops with a clear message on bad data."""
    reader = csv.reader(io.StringIO(read_questions_text(source)))
    header = next(reader)
    if header != QUESTION_HEADER:
        raise SystemExit(f"Unexpected header {header}; expected {QUESTION_HEADER}")

    questions: list[Question] = []
    seen: set[str] = set()
    type_of_pattern: dict[str, str] = {}
    problems: list[str] = []
    for line_no, row in enumerate(reader, start=2):
        if len(row) != 4:
            problems.append(f"line {line_no}: {len(row)} fields")
            continue
        text, pid, tid, meaning = (c.strip() for c in row)
        if not text or "\n" in text or "\r" in text:
            problems.append(f"line {line_no}: empty or multi-line question")
            continue
        if text in seen:
            problems.append(f"line {line_no}: duplicate question {text!r}")
            continue
        if type_of_pattern.setdefault(pid, tid) != tid:
            problems.append(f"line {line_no}: pattern {pid} has two types")
            continue
        try:
            label = json.loads(meaning)
        except json.JSONDecodeError:
            problems.append(f"line {line_no}: 意味 is not valid JSON")
            continue
        if label.get("パターン") != pid or label.get("質問タイプ") != tid:
            problems.append(f"line {line_no}: 意味 does not match its pattern/type columns")
            continue
        seen.add(text)
        questions.append(Question(text, pid, tid, meaning))

    if problems:
        print(f"{len(problems)} problem(s) in the corpus; first 10:")
        for p in problems[:10]:
            print("  ", p)
        raise SystemExit("Fix the corpus (or regenerate it with the team's generator) and run again.")
    return questions


def load_patterns(path: Path) -> dict[str, PatternInfo]:
    """Load patterns.csv into a dict keyed by pattern ID."""
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    return {
        r["パターンID"]: PatternInfo(r["パターンID"], r["質問タイプID"], r["質問タイプ"], r["文体"], r["文型"])
        for r in rows
    }


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


# ================================================================ splitting
_NEAR_DUP_STRIP = re.compile(r"[\s。、，．,.!！?？・「」『』（）()〜~ー－\-]")


def near_dup_key(text: str) -> str:
    """Text with punctuation, spaces and long-vowel marks removed (after NFKC)."""
    return _NEAR_DUP_STRIP.sub("", unicodedata.normalize("NFKC", text))


def choose_held_out_patterns(
    questions: list[Question], patterns: dict[str, PatternInfo], cfg: SplitConfig
) -> dict[str, str]:
    """Pick whole patterns for val and test. Returns pattern_id -> split for every non-empty pattern.

    Rules (all deterministic for a given seed):
    - only types with at least cfg.min_type_lines questions and at least 3 non-empty patterns
      may donate a pattern, so every type keeps at least 2 phrasings in train;
    - each chosen type donates exactly ONE pattern, and val types and test types are disjoint,
      so the held-out sets cover as many different types as possible;
    - the donated pattern must have at least cfg.min_pattern_lines questions and at most
      cfg.max_donate_share of its type's questions, so train keeps most of every type.
    """
    rng = random.Random(cfg.seed)
    per_pattern = Counter(q.pattern_id for q in questions)
    per_type = Counter(q.type_id for q in questions)
    patterns_of_type: dict[str, list[str]] = defaultdict(list)
    for pid in sorted(per_pattern):
        patterns_of_type[patterns[pid].type_id].append(pid)

    def candidates_of(tid: str) -> list[str]:
        return [
            p for p in patterns_of_type[tid]
            if cfg.min_pattern_lines <= per_pattern[p] <= cfg.max_donate_share * per_type[tid]
        ]

    eligible: list[str] = []
    for tid in sorted(patterns_of_type):
        candidates = candidates_of(tid)
        if per_type[tid] >= cfg.min_type_lines and len(patterns_of_type[tid]) >= 3 and candidates:
            eligible.append(tid)

    n_patterns = len(per_pattern)
    n_val = round(cfg.val_fraction * n_patterns)
    n_test = round(cfg.test_fraction * n_patterns)
    if n_val + n_test > len(eligible):
        print(f"WARNING: only {len(eligible)} eligible types; reducing held-out patterns.")
        n_val = len(eligible) // 2
        n_test = len(eligible) - n_val

    rng.shuffle(eligible)
    assignment = {pid: "train" for pid in per_pattern}
    for split, chosen_types in (("val", eligible[:n_val]), ("test", eligible[n_val:n_val + n_test])):
        for tid in sorted(chosen_types):
            assignment[rng.choice(candidates_of(tid))] = split
    return assignment


def build_splits(questions: list[Question], assignment: dict[str, str], cfg: SplitConfig) -> SplitResult:
    """Apply the pattern assignment, remove near-duplicates, cap eval patterns, oversample train."""
    rng = random.Random(cfg.seed + 1)
    by_split: dict[str, list[Question]] = {s: [] for s in SPLITS}
    for q in questions:
        by_split[assignment[q.pattern_id]].append(q)

    # Near-duplicate guard: train wins; val loses to train; test loses to train and val.
    dropped_near_dup: dict[str, int] = {}
    blocked = {near_dup_key(q.text) for q in by_split["train"]}
    for split in ("val", "test"):
        kept = [q for q in by_split[split] if near_dup_key(q.text) not in blocked]
        dropped_near_dup[split] = len(by_split[split]) - len(kept)
        by_split[split] = kept
        blocked |= {near_dup_key(q.text) for q in kept}

    # Cap each held-out pattern (random sample, seeded). Dropped lines are NOT moved to train.
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

    # Light oversampling of small types, train only.
    train_unique = list(by_split["train"])
    per_type = Counter(q.type_id for q in train_unique)
    repeat_factor = {
        tid: min(cfg.max_repeat, max(1, math.ceil(cfg.oversample_target / n))) for tid, n in per_type.items()
    }
    train: list[Question] = []
    for q in train_unique:
        train.extend([q] * repeat_factor[q.type_id])
    by_split["train"] = train

    for split in SPLITS:
        rng.shuffle(by_split[split])
    return SplitResult(assignment, by_split, train_unique, repeat_factor, dropped_near_dup, dropped_by_cap)


# =============================================================== statistics
def script_class(ch: str) -> str:
    """Script class of one character, matching the planned Day 6 pre-tokenizer."""
    o = ord(ch)
    if 0x3041 <= o <= 0x309F:
        return "hiragana"
    if (0x30A1 <= o <= 0x30FA) or o in (0x30FC, 0x30FD, 0x30FE) or 0x31F0 <= o <= 0x31FF or 0xFF66 <= o <= 0xFF9F:
        return "katakana"
    if 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF or o in (0x3005, 0x3006, 0x30F6):
        return "kanji"
    if ch.isascii() and ch.isalpha() or 0xFF21 <= o <= 0xFF3A or 0xFF41 <= o <= 0xFF5A:
        return "latin"
    if ch.isdigit():
        return "digit"
    if ch.isspace():
        return "space"
    return "other"


def pretoken_chunks(text: str) -> int:
    """Number of same-script runs. A byte-level BPE with this pre-tokenizer cannot merge across
    runs, so this is a hard LOWER bound on the token count. UTF-8 bytes are the UPPER bound."""
    chunks = 0
    previous = ""
    for ch in text:
        cls = script_class(ch)
        if cls != previous or cls == "other":
            chunks += 1
        previous = cls
    return chunks


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


def coverage(texts: Iterable[str], words: list[str]) -> int:
    """How many of `words` occur at least once in the texts (substring match)."""
    blob = "\n".join(texts)
    return sum(1 for w in words if w in blob)


# ================================================================== writing
def write_lines(path: Path, qs: list[Question]) -> None:
    """One question per line, UTF-8 without BOM, LF line endings (what the tokenizer reads)."""
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for q in qs:
            f.write(q.text + "\n")


def write_labels(path: Path, qs: list[Question], split: str) -> None:
    """Internal label file (not training text). Same order as the .txt file."""
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["質問", "パターンID", "質問タイプID", "split", "意味"])
        for q in qs:
            w.writerow([q.text, q.pattern_id, q.type_id, split, q.meaning])


def fmt(n: int | float) -> str:
    return f"{n:,}" if isinstance(n, int) else f"{n:,.2f}"


def write_data_card(
    path: Path,
    stats: dict[str, dict[str, int]],
    per_type_rows: list[list[str]],
    result: SplitResult,
    patterns: dict[str, PatternInfo],
    empty_patterns: list[str],
    cov: dict[str, dict[str, int]],
    n_terms: int,
    n_variants: int,
    cfg: SplitConfig,
    source_name: str,
    source_sha: str,
) -> None:
    """docs/DATA_CARD.md. Every number comes from this run; nothing is typed by hand."""
    held = {s: sorted(p for p, sp in result.assignment.items() if sp == s) for s in ("val", "test")}
    style_counts = {s: Counter(patterns[p].style for p in held[s]) for s in held}
    lines: list[str] = [
        "# Data card: Model A training corpus",
        "",
        f"Generated by `scripts/day5_build_corpus.py` ({SCRIPT_VERSION}) on {datetime.now():%Y-%m-%d %H:%M}.",
        "All numbers below come from that run. Re-run the script to refresh them.",
        "",
        "## 1. What the corpus is",
        "",
        "Short Japanese questions about a company's employee data (departments, positions, pay,",
        "allowances, attendance, evaluations, dates), written the way staff would ask them.",
        "Model A is a text-continuation model, so it learns to continue a few words into a natural question.",
        "",
        "## 2. Sources and license",
        "",
        "| Source | Made by | Use |",
        "| --- | --- | --- |",
        "| Glossary (用語集) | the team, original wording | words that may fill the patterns |",
        "| Synonym dictionary (同義語辞書) | the team, original wording | variant words (for example ボーナス for 賞与) |",
        "| Question patterns (src_*.txt, patterns.csv) | the team | sentence frames with slots |",
        "| Sample employee table, 50 rows | the team | names, IDs and realistic value ranges for the slots |",
        "| Generator (make_questions.py) | the team | fills the patterns, removes duplicates |",
        "",
        "No outside text is used: no Wikipedia, no books, no web pages, no laws.",
        "All training text was created by the team's own generator from the team's own files.",
        "",
        f"Input file: `{source_name}`, SHA-256 `{source_sha}`.",
        "",
        "## 3. How it was split",
        "",
        "- **By pattern, never by line.** A pattern is one sentence frame (one phrasing).",
        "  Lines from one pattern are nearly identical, so a random line split would put almost the",
        "  same sentence in train and test and make the scores look better than they are.",
        f"- {len(held['val'])} patterns are held out for **validation** and {len(held['test'])} for **test**,",
        "  each from a different question type. The model never sees these phrasings in training,",
        "  but it does see other phrasings of the same question types.",
        f"- A type may donate a pattern only if it has at least {cfg.min_type_lines} questions and at least",
        "  3 phrasings, so every type keeps at least 2 phrasings in train. The held-out pattern may hold",
        f"  at most {cfg.max_donate_share:.0%} of its type's questions, so train keeps most of every type.",
        f"- Near-duplicates (same text after removing punctuation and spaces) were removed from the",
        f"  held-out sets: validation {result.dropped_near_dup['val']}, test {result.dropped_near_dup['test']}.",
        f"- Each held-out pattern is capped at {cfg.eval_cap_per_pattern} questions; lines over the cap",
        f"  are dropped, not moved to train (validation {result.dropped_by_cap['val']}, test {result.dropped_by_cap['test']}).",
        f"- Seed {cfg.seed}. The full list of patterns per split is in `data/splits/split_manifest.json`.",
        f"- Held-out styles: validation {dict(style_counts['val'])}, test {dict(style_counts['test'])}.",
        "",
        "## 4. Balancing",
        "",
        f"Some question types can only produce a few distinct questions. In **train only**, each small type",
        f"is repeated up to about {cfg.oversample_target} lines, at most {cfg.max_repeat} copies.",
        "Validation and test are never repeated.",
        "",
        "## 5. Size",
        "",
        "| Split | Lines | Unique lines | Characters | UTF-8 bytes | Token lower bound | Patterns | Types | Longest line |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for s in SPLITS:
        st = stats[s]
        lines.append(
            f"| {s} | {fmt(st['lines'])} | {fmt(st['unique_lines'])} | {fmt(st['characters'])} | {fmt(st['utf8_bytes'])} "
            f"| {fmt(st['token_lower_bound'])} | {st['patterns']} | {st['types']} | {st['max_chars']} |"
        )
    lines += [
        "",
        "**Tokens:** the real count is measured on Day 7 with our own tokenizer. Until then it is bounded:",
        "at least the token lower bound (number of same-script runs, which a byte-level BPE with a",
        "script-class pre-tokenizer cannot merge across) and at most the UTF-8 byte count.",
        "",
        "## 6. Per question type",
        "",
        "| Type | Name | Train unique | Copies | Train lines | Validation | Test |",
        "| --- | --- | --- | --- | --- | --- | --- |",
        *("| " + " | ".join(r) + " |" for r in per_type_rows),
        "",
        "## 7. Vocabulary coverage",
        "",
        f"Glossary standard terms: {n_terms}. Synonym variants: {n_variants}.",
        "",
        "| Split | Standard terms that appear | Variants that appear |",
        "| --- | --- | --- |",
        *(f"| {s} | {cov[s]['terms']} | {cov[s]['variants']} |" for s in SPLITS),
        "",
        "Many glossary terms are general company words that the questions do not use, so coverage",
        "below 100% is expected. Coverage is a substring check, so very short terms can match inside",
        "longer words.",
        "",
        "## 8. Known limits and biases",
        "",
        "- Narrow domain: questions about one kind of employee data only, no statements or general text.",
        "- Template-made: phrasing variety is limited to the team's patterns; real users will say things",
        "  the patterns do not cover. Held-out patterns measure only part of this gap.",
        "- People: names and IDs come from a 50-row sample table, so the same names repeat often.",
        "- Uneven types: some types have very few distinct questions; they are oversampled a little",
        "  and may still be learned less well.",
        f"- Patterns with no questions after duplicate removal: {', '.join(empty_patterns) or 'none'}"
        " (their sentences are identical to sentences of an earlier pattern).",
        "- Ambiguous everyday words (for example 給料) are kept out of the patterns until the team",
        "  decides what they mean.",
        "",
        "## 9. Reproduce",
        "",
        "```powershell",
        ".venv\\Scripts\\python.exe scripts\\day5_build_corpus.py --machine home",
        "```",
        "",
        "The output hashes in `data/splits/split_manifest.json` must match on every machine.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def append_run_log(log: Path, machine: str, stats: dict[str, dict[str, int]], manifest_rel: str, cfg: SplitConfig,
                   n_val: int, n_test: int) -> None:
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


# ===================================================================== main
def parse_args() -> argparse.Namespace:
    d = SplitConfig()
    ap = argparse.ArgumentParser(description="Day 5: split the question corpus by pattern and write the data card.")
    ap.add_argument("--machine", choices=["school", "home"], required=True, help="label for run_log.md")
    ap.add_argument("--source", type=Path, default=REPO_ROOT / "data/source/questions_corpus.zip",
                    help="questions_corpus.zip or questions.csv")
    ap.add_argument("--patterns", type=Path, default=REPO_ROOT / "data/source/patterns.csv")
    ap.add_argument("--glossary", type=Path, default=REPO_ROOT / "data/source/用語集.csv")
    ap.add_argument("--synonyms", type=Path, default=REPO_ROOT / "data/source/同義語辞書.csv")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data/corpus")
    ap.add_argument("--manifest", type=Path, default=REPO_ROOT / "data/splits/split_manifest.json")
    ap.add_argument("--data-card", type=Path, default=REPO_ROOT / "docs/DATA_CARD.md")
    ap.add_argument("--run-log", type=Path, default=REPO_ROOT / "run_log.md")
    ap.add_argument("--no-log", action="store_true", help="do not append to run_log.md (for dry runs)")
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


def main() -> int:
    # A Japanese error message must never crash a console that is not UTF-8 (for example cp932).
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    a = parse_args()
    for name in ("source", "patterns", "glossary", "synonyms", "out_dir", "manifest", "data_card", "run_log"):
        setattr(a, name, getattr(a, name).resolve())  # so relative paths from the command line also work
    cfg = SplitConfig(a.seed, a.val_fraction, a.test_fraction, a.min_type_lines, a.min_pattern_lines,
                      a.max_donate_share, a.eval_cap, a.oversample_target, a.max_repeat)
    for p in (a.source, a.patterns):
        if not p.exists():
            raise SystemExit(f"Missing input: {p}")

    print(f"=== Day 5 corpus split | machine={a.machine} | seed={cfg.seed} ===")
    questions = load_questions(a.source)
    patterns = load_patterns(a.patterns)
    unknown = sorted({q.pattern_id for q in questions} - patterns.keys())
    if unknown:
        raise SystemExit(f"Pattern IDs not in patterns.csv: {unknown[:10]}")
    wrong_type = sorted({q.pattern_id for q in questions if patterns[q.pattern_id].type_id != q.type_id})
    if wrong_type:
        raise SystemExit(f"Type in corpus differs from patterns.csv for: {wrong_type[:10]}")
    used = {q.pattern_id for q in questions}
    empty_patterns = sorted(set(patterns) - used)
    print(f"Loaded {len(questions):,} questions, {len(used)} patterns with questions, "
          f"{len({q.type_id for q in questions})} types.")
    if empty_patterns:
        print(f"Patterns with no questions (all their sentences already came from another pattern): {empty_patterns}")

    assignment = choose_held_out_patterns(questions, patterns, cfg)
    result = build_splits(questions, assignment, cfg)

    # ---- write outputs
    a.out_dir.mkdir(parents=True, exist_ok=True)
    a.manifest.parent.mkdir(parents=True, exist_ok=True)
    a.data_card.parent.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    for s in SPLITS:
        txt = a.out_dir / f"{s}.txt"
        lab = a.out_dir / f"{s}_labels.csv"
        write_lines(txt, result.lines[s])
        write_labels(lab, result.lines[s], s)
        outputs[str(txt.relative_to(REPO_ROOT)).replace("\\", "/")] = sha256_of(txt)
        outputs[str(lab.relative_to(REPO_ROOT)).replace("\\", "/")] = sha256_of(lab)

    stats = {s: split_stats(result.lines[s]) for s in SPLITS}
    stats["train_unique"] = split_stats(result.train_unique)

    # per-type table
    tr_u = Counter(q.type_id for q in result.train_unique)
    tr = Counter(q.type_id for q in result.lines["train"])
    va = Counter(q.type_id for q in result.lines["val"])
    te = Counter(q.type_id for q in result.lines["test"])
    type_names = {p.type_id: p.type_name for p in patterns.values()}
    per_type_rows = [
        [t, type_names.get(t, ""), str(tr_u[t]), str(result.repeat_factor.get(t, 1)), str(tr[t]), str(va[t]), str(te[t])]
        for t in sorted({q.type_id for q in questions})
    ]

    terms, variants = load_terms(a.glossary, a.synonyms)
    cov = {
        s: {"terms": coverage((q.text for q in result.lines[s]), terms),
            "variants": coverage((q.text for q in result.lines[s]), variants)}
        for s in SPLITS
    }
    if not terms:
        print("NOTE: glossary/synonym files not found; the coverage section will show 0.")

    held = {s: sorted(p for p, sp in assignment.items() if sp == s) for s in SPLITS}
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
    a.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    write_data_card(a.data_card, stats, per_type_rows, result, patterns, empty_patterns, cov,
                    len(terms), len(variants), cfg, a.source.name, manifest["inputs"][a.source.name])

    # ---- sanity checks on the written result (fail loudly if the split leaked)
    train_keys = {near_dup_key(q.text) for q in result.train_unique}
    leak = sum(1 for s in ("val", "test") for q in result.lines[s] if near_dup_key(q.text) in train_keys)
    overlap = set(held["val"]) & set(held["test"]) | (set(held["val"]) | set(held["test"])) & set(held["train"])
    if leak or overlap:
        raise SystemExit(f"Split check FAILED: {leak} leaked lines, {len(overlap)} shared patterns.")

    # ---- console summary
    print()
    print(f"{'split':<6} {'lines':>9} {'unique':>9} {'chars':>11} {'token bounds':>25} {'patterns':>9} {'types':>6}")
    for s in SPLITS:
        st = stats[s]
        bounds = f"{st['token_lower_bound']:,} - {st['utf8_bytes']:,}"
        print(f"{s:<6} {st['lines']:>9,} {st['unique_lines']:>9,} {st['characters']:>11,} {bounds:>25} "
              f"{st['patterns']:>9} {st['types']:>6}")
    print(f"Near-duplicates removed: {result.dropped_near_dup}; over eval cap: {result.dropped_by_cap}")
    oversampled = {t: f for t, f in sorted(result.repeat_factor.items()) if f > 1}
    print(f"Oversampled types (copies): {oversampled}")
    print(f"Coverage {cov}")
    print("Split check: PASS (no shared patterns, no near-duplicate leaks)")
    print(f"Wrote {a.out_dir}, {a.manifest}, {a.data_card}")

    if not a.no_log:
        append_run_log(a.run_log, a.machine, stats, str(a.manifest.relative_to(REPO_ROOT)).replace("\\", "/"),
                       cfg, len(held["val"]), len(held["test"]))
        print(f"Appended an entry to {a.run_log.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
